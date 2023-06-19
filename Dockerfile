FROM fame-docker-registary-registry.cn-shanghai.cr.aliyuncs.com/fame/ai-openvino:v4
WORKDIR /usr/src/app
COPY . .
CMD [ "python", "server.py" ]
