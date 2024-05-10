FROM  nvcr.io/nvidia/pytorch:22.10-py3
# maybe we also have a requirements.txt file
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt
#COPY ./SOP ./SOP
WORKDIR KolektorSDD2
#ENTRYPOINT ["python"]
#CMD ["train.py","-c", "config_cifar100.json", "--lr_u","1", "--lr_v" ,"10","--percent","0.5"]

