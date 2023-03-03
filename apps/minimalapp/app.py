#Flaskクラスをインポートする
from flask import Flask, render_template, url_for, current_app, g, request, redirect

#Flaskクラスをインスタンス化する
app = Flask(__name__)

#URLと実行する関数をマッピングする
@app.route("/")
def index():
    return "Hello, Flaskbook!"

@app.route("/hello", methods=["GET"],endpoint="hello-endpoint")
def hello():
    return "Hello, world!"

@app.route("/hello/<name>")
def hello2(name):
    return f"Hello,{name}!"

@app.route("/name/<name>")
def show_name(name):
    return render_template("index.html",name=name)

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/contact/complete", method=["GET","POST"])
def contact_complete():
    #POSTの場合のみ以下を実行
    if request.method == "POST":
    #メールを送る(最後に実装)
    
    #contactエンドポイントへリダイレクトする
        return redirect(url_for("contact_complete"))
    return render_template("contact_complete.html")

with app.test_request_context():
    #/
    print(url_for("index"))
    #/hello/world
    print(url_for("hello-endpoint", name="world"))
    #/name/ichiro?page=1
    print(url_for("show_name", name="ichiro", page="1"))