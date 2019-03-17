
import csv

usr_name, pass_word = 'heliusha@126.com', '123456'

with open('my_text_file.csv','w') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(["User Name", "Password"])
    thewriter.writerow([usr_name, pass_word])

with open('my_text_file.csv','a') as f:
    thewriter = csv.writer(f)
    thewriter.writerow([usr_name, pass_word])
