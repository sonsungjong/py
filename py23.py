# # 105page
# n = 10
# while n > 0:
#     print(n,"번 했습니다.")
# #     n -= 1

# # 111page (1)
# '''
# 숫자 1개를 입력받고
# 그 숫자만큼 Hello 하기
# 단, 0 이하의 숫자를 입력할 경우
# '잘못된 숫자'입니다만 해주기
# '''
# 숫자한개 = int(input("숫자 한개 입력>"))
# 기준 = 0
# if 숫자한개 <= 0:
#     print('잘못된 숫자 입력')
# else:
#     while 기준 < 숫자한개:
#         print(기준+1,'번째 hello')
#         기준+=1

# 111page (2)
'''
1~100까지 숫자 중에서
7의 배수만 출력하기
(반복문 사용)

(1) 1 ~ 100 출력
(2) if문을 써서 7의 배수만 출력되게 바꾸기
'''
flag = 1
while flag < 101:
    if flag % 7 == 0:
        print(flag)
    flag = flag + 1         # flag += 1


# 110page (2중반복문)
