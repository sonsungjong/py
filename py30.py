리스트 = ['h','e','l','l','o']
문자열 = 'hello'

# 잘라내기
print(리스트[1:4])      # ['e','l','l']
print(문자열[1:4])      # 'ell'
print(리스트[-1])       # 'o'
print(문자열[-1])       # 'o'

글자 = "3학년 5반 10번"
print(글자[7:10])
print(글자[7:])

# page 99 (4번)
# 입력예시 : 237가1234
# 입력예시 : 33바3388
# 입력예시 : 138마7777

# hint 입력값[-1], 입력값[6:]
차량번호 = input("차량번호를 입력하세요>>>")
if int(차량번호[-1]) % 2 == 0:
    print("운행 가능")
else:
    print("운행 불가")
