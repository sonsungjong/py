# # 1
# # 100~90 : A, 89~80 : B, ....
# 점수 = int(input("점수를 입력하세요>>>"))
# if 점수 < 60:
#     print('점수는',점수,'점이고, 학점은 F학점입니다.')
# elif 점수 < 70:
#     print('점수는',점수,'점이고, 학점은 D학점입니다.')
# elif 점수 < 80:
#     print('점수는',점수,'점이고, 학점은 C학점입니다.')
# elif 점수 < 90:
#     print('점수는',점수,'점이고, 학점은 B학점입니다.')
# elif 점수 <= 100:
#     print('점수는',점수,'점이고, 학점은 A학점입니다.')
# else:
#     print('?')

# 2
# 정수 = int(input('정수를 입력하세요>>>'))
# if 정수 % 3 == 0:
#     print(정수,'는 3의 배수입니다.')
# else:
#     print(정수,'는 3의 배수가 아닙니다.')

# 3
while True:
    정수1 = int(input('정수1 입력>>>'))
    정수2 = int(input('정수2 입력>>>'))
    정수3 = int(input('정수3 입력>>>'))

    if 정수1 >= 정수2 and 정수1 >= 정수3:
        print('가장 큰 수는',정수1,'입니다.')
    elif 정수2 >= 정수1 and 정수2 >= 정수3:
        print('가장 큰 수는',정수2,'입니다.')
    else:
        print('가장 큰 수는',정수3,'입니다.')