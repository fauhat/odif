import requests

"""
api_url = "https://jsonplaceholder.typicode.com/todos/"
response = requests.get(api_url)
user_list = response.json()
user = user_list[0]
#print("userId: ", response_json['userId'])
print("response.statuscode: ",response.status_code)
print("user_list: ",user_list)
print("type(user_list): ",type(user_list))
print("type(user): ",type(user))
print("user: ", user)
print("userId: ", user['userId'])
"""
api_url = "http://54.174.31.133/"
response = requests.get(api_url, headers={"Host": "www.difserver1.com"})
response_json = response.json()
print("response.status", response.status_code)
print("response-json", response_json)



