from lib import *
import pickle
import zlib
import sys
import base64
import os
from cryptography.fernet import Fernet

def save_encrypted_object(obj, filename, key):
    # 1. 객체 직렬화 및 압축
    serialized_data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    compressed_data = zlib.compress(serialized_data, level=9)
    
    # 2. 암호화
    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(compressed_data)
    
    # 3. 난독화 데이터 생성
    p_len_val = os.urandom(1)[0] % 32 + 10
    prefix = os.urandom(p_len_val)
    suffix = os.urandom(os.urandom(1)[0] % 32 + 10)
    
    # [헤더 구성] 
    # 2바이트: 접두사(Prefix) 길이
    # 4바이트: 실제 암호화 데이터(Fernet Token) 길이
    header = p_len_val.to_bytes(2, byteorder='big') + len(encrypted_data).to_bytes(4, byteorder='big')
    
    # 최종 바이너리 구성
    final_data = header + prefix + encrypted_data + suffix
    
    with open(filename, 'wb') as f:
        f.write(final_data)

def main(fn, text):
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()
    if error:
        print(error.as_string())
        return

    parser = Parser(tokens)
    res = parser.parse()
    if res.error:
        print(res.error.as_string())
        return
    
    raw_key = b'TriglavSecrateKey123456789012345'
    secret_key = base64.urlsafe_b64encode(raw_key)
    
    save_encrypted_object(res.node, sys.argv[2], secret_key)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: triglavm <input> <output>")
        sys.exit(1)
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        main(sys.argv[1], f.read())
