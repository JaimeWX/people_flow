import os
import redis

RTSP_IP = os.environ["RTSP_IP"] if "RTSP_IP" in os.environ else '192.168.82.117'

const_redis_host = os.environ["REDIS_HOST"] if "REDIS_HOST" in os.environ else "redis"  
const_redis_username = os.environ["REDIS_USERNAME"] if "REDIS_USERNAME" in os.environ else "admin"
const_redis_password = os.environ["REDIS_PASSWORD"] if "REDIS_PASSWORD" in os.environ else "redis@fame!2019"   
pool = redis.ConnectionPool(host=const_redis_host,password=const_redis_password, port=6379, decode_responses=True)  
client = redis.Redis(connection_pool=pool)

client.hset('test_rtsp','14:00',100)
print(client.hgetall(RTSP_IP))