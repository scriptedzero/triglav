#!/usr/bin/python3

import lib
import sys
import os


def main():
	if len(sys.argv) > 1:
		dyr, fn = os.path.split(sys.argv[1])
		try:
			os.chdir(dyr)
		except:
			pass
		with open(fn, 'r') as f:
			code = f.read()
		_, error = lib.run(fn, code)
		if error:
			print(error.as_string(), file=sys.stderr)
			exit(1)

	while True:
		try:
			text = input('repl@triglav > ')
			if text.strip() == '':
				continue

			result, error = lib.run('<stdin>', text)

			if error:
				print(error.as_string(), file=sys.stderr)
			elif result:
				real_result = result.elements[0]
				if len(result.elements) != 1:
					real_result = result
				print(repr(real_result))
				lib.global_symbol_table.set('_', real_result)
		except KeyboardInterrupt:
			sys.exit(1)


if __name__ == '__main__':
	main()
