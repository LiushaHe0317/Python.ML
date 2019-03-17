
import tkinter as tk
from tkinter import messagebox as mgb
import pickle

window = tk.Tk()
window.title('Welcome to My System')
window.geometry('500x300')

# welcome image
canvas = tk.Canvas(window, height = 200, width = 500)
image_file = tk.PhotoImage(file = "Welcome.gif")
show_image = canvas.create_image(0, 0, anchor = 'nw', 
                                image = image_file)
canvas.pack(side = 'top')

# labels
tk.Label(window,text = 'User Name').place(x = 50, y = 150)
tk.Label(window, text = 'Password').place(x = 50, y = 190)

# entry
var_usr_name = tk.StringVar()
var_usr_name.set('example@python.com')
entry_isr_name = tk.Entry(window, textvariable = var_usr_name)
entry_isr_name.place(x = 160, y = 150)
var_pass_word = tk.StringVar()
entry_isr_name = tk.Entry(window, textvariable = var_pass_word, show = '*')
entry_isr_name.place(x = 160, y = 190)

def usr_login():
    usr_name = var_usr_name.get()
    usr_pwd = var_pass_word.get()
    
    try:
        with open('usrs_info.pickle','rb') as usrs_file:
            usrs_info = pickle.load(usrs_file)
    except FileNotFoundError:
        with open('usrs_info.pickle','wb') as usrs_file:
            usrs_info = ['admin','admin']
            pickle.dump(usrs_info, usrs_file)
    
    if usr_name in usrs_info:
        if usr_pwd == usrs_info[usrs_info.index(usr_name) + 1]:
            mgb.showinfo(title = 'Welcome', message = 'Welcome back ' + usr_name)
        else:
            mgb.showerror('Error', 'your password is incorrect, please try again.')
    else:
        is_sign_up = mgb.showinfo('Info','You have not signed up yet, sign up Today?')
    
        if is_sign_up:
            usr_sign_up()
    
def usr_sign_up():
    def sign_up_confirm():
        nn = new_name.get()
        np = new_pwd.get()
        npf = new_pwd_confirm.get()
        
        with open('usrs_info.pickle','rb') as usrs_file:
            exist_usr_info = pickle.load(usrs_file)
        
        if np != npf:
            mgb.showerror('Error', 'Password and confirm password must be the same ')
        elif nn in exist_usr_info:
            mgb.showerror('Error', 'The user name has been registered')
        else:
            exist_usr_info = [nn, np]
            
            with open('usrs_info.pickle','wb') as usrs_file:
                pickle.dump(exist_usr_info, usrs_file)
                
            mgb.showinfo('Wlcome', 'You have successfully signed up')
            window_sign_up.destroy()

    window_sign_up = tk.Toplevel(window)
    window_sign_up.geometry('350x200')
    window_sign_up.title('Sign Up Window')
    
    # user name
    new_name = tk.StringVar()
    new_name.set('example@python.com')
    tk.Label(window_sign_up, text = 'User Name').place(x = 10, y = 10)
    entry_new_name = tk.Entry(window_sign_up, textvariable = new_name)
    entry_new_name.place(x = 150, y = 10)
    
    # password
    new_pwd = tk.StringVar()
    tk.Label(window_sign_up, text = 'Password').place(x = 10, y = 60)
    entry_new_pwd = tk.Entry(window_sign_up, textvariable = new_pwd, show = '*')
    entry_new_pwd.place(x = 150, y = 60)
    
    # confirm password
    new_pwd_confirm = tk.StringVar()
    tk.Label(window_sign_up, text = 'Confirm Password').place(x = 10, y = 110)
    entry_new_pwd = tk.Entry(window_sign_up, textvariable = new_pwd_confirm, show = '*')
    entry_new_pwd.place(x = 150, y = 110)
    
    btn_sign_up_confirm = tk.Button(window_sign_up, text = 'Sign Up', command = sign_up_confirm)
    btn_sign_up_confirm.place(x = 150, y = 150)

# login button
btn_login = tk.Button(window, text = 'Login', command = usr_login)
btn_login.place(x = 250, y = 230) 

# sign-up button
btn_sign_up = tk.Button(window, text = 'Sign Up', command = usr_sign_up)
btn_sign_up.place(x = 350, y = 230)

window.mainloop()
