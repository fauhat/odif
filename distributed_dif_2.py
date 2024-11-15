import time
import requests
import requests_async
import asyncio
import aiohttp

instance1_url = "http://54.81.242.1/dif-run-time/?ensemblesize=100"
instance2_url = "http://54.85.163.17/dif-run-time/?ensemblesize=100"
#instance3_url = "http://54.174.31.133/dif-run-time/?ensemblesize=16"

async def main():
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = []

        # task = asyncio.ensure_future(call_dif_api(session, instance1_url)) 
        # tasks.append(task)
        
        task = asyncio.ensure_future(call_dif_api(session, instance2_url)) 
        tasks.append(task)

        responses = await asyncio.gather(*tasks)
    
    end_time = time.time()
    runtime = end_time - start_time
    print("Responses: ", responses)
    print("Run Time: ", runtime)

async def call_dif_api(session, url):
    async with session.get(url) as response:
        response_json = await response.json()
        return response_json

asyncio.run(main())