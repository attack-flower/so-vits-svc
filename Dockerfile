FROM python:3.10.11

RUN mkdir /so-vits-svc
WORKDIR /so-vits-svc

COPY . .

RUN pip install -r requirements.txt

RUN apt update && apt -y install -qq aria2
RUN aria2c --console-log-level=error -c -x 16 -k 1M -s 16 https://ibm.ent.box.com/shared/static/z1wgl1stco8ffooyatzdwsqn2psd9lrr -o checkpoint_best_legacy_500.pt -d /so-vits-svc/hubert

EXPOSE 7860

CMD ["python", "myapp.py"]
