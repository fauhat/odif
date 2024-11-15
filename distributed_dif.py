
import time
import requests
import requests_async
import asyncio

instance1_url = "http://3.84.67.125/dif-run-time/?ensemblesize=50"
instance2_url = "http://54.80.153.130/dif-run-time/?ensemblesize=50"
#instance3_url = "http://54.174.31.133/dif-run-time/?ensemblesize=16"

async def run():
    start_time = time.time()

    #response = requests.get(instance1_url)
    #response = await requests_async.get(instance1_url)
    response1 = requests_async.get(instance1_url)
    #print("remote api response: ", response.content)

    
    #response = requests.get(instance2_url)
    #response = await requests_async.get(instance2_url)
    response2 = requests_async.get(instance2_url)
    #print("remote api response: ", response.content)

    a_response1 = await response1
    a_response2 = await response2

    """
    response = requests.get(instance3_url)
    print("remote api response: ", response.content)
    """
    end_time = time.time()
    runtime = end_time - start_time
    print("runtime: ", runtime)

asyncio.run(run())
