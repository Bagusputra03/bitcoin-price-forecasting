import hashlib, secrets, getpass
password=getpass.getpass('Password baru: ')
salt=secrets.token_hex(16); iterations=260000
digest=hashlib.pbkdf2_hmac('sha256',password.encode(),salt.encode(),iterations).hex()
print(f'pbkdf2_sha256${iterations}${salt}${digest}')
