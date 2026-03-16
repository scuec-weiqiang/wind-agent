from chat import ask,ask_stream

while True:

    user = input("You: ")

    reply = ask_stream(user)

    print("AI:", reply)