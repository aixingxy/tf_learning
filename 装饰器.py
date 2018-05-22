# -*- coding:utf-8 -*-
# def log(func):
#     def wrapper(*arg, **kw):
#         print("start {}".format(func.__name__))
#         return func(*arg, **kw)
#     return wrapper
#
# @log
# def fun_a(arg):
#     pass
#
# @log
# def fun_b(arg):
#     pass
#
# @log
# def fun_c(arg):
#     pass
#
# fun_a(1)
# fun_b(2)
# fun_c(3)

from flask import Flask
app = Flask(__name__)
@app.route('/')
def index():
    return "Index Page"

@app.route('/hello')
def hello():
    return "Hello World"

if __name__ == "__main__":
    app.run()

