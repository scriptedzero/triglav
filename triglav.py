from lib import *
import pickle
import zlib
import sys
import base64
from cryptography.fernet import Fernet


def load_encrypted_object(filename, key):
	try:
		with open(filename, 'rb') as f:
			raw_data = f.read()

		if len(raw_data) < 6:
			return None

		# 1. 헤더 분석 (총 6바이트)
		prefix_len = int.from_bytes(raw_data[:2], byteorder='big')
		content_len = int.from_bytes(raw_data[2:6], byteorder='big')

		# 2. 암호화 데이터 시작점과 끝점 계산
		start_pos = 6 + prefix_len
		end_pos = start_pos + content_len

		# 정확히 암호화된 구간만 슬라이싱 (중요!)
		encrypted_content = raw_data[start_pos:end_pos]

		# 3. 복호화 및 복원
		fernet = Fernet(key)
		decrypted_data = fernet.decrypt(encrypted_content)
		return pickle.loads(zlib.decompress(decrypted_data))
	except Exception:
		sys.exit(1)


def main():
	if len(sys.argv) < 2:
		sys.exit(1)

	raw_key = b'TriglavSecrateKey123456789012345'
	secret_key = base64.urlsafe_b64encode(raw_key)

	ast_node = load_encrypted_object(sys.argv[1], secret_key)
	if not ast_node:
		sys.exit(1)

	interpreter = Interpreter()
	context = Context('<program>')
	try:
		context.symbol_table = global_symbol_table
	except NameError:
		pass

	result = interpreter.visit(ast_node, context)
	if result.error:
		sys.exit(1)

	ret = result.func_return_value
	if ret and isinstance(ret, Number):
		sys.exit(int(ret.value))


if __name__ == '__main__':
	main()
