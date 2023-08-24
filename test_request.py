import requests

app_url = "http://127.0.0.1:5000/get_response"

user_input = input("You: ")

response = requests.post(app_url, data={"user_input": user_input})

if response.status_code == 200:
    bot_response = response.text
    print("Bot:", bot_response)
else:
    print("Failed to get a response from the bot.")
