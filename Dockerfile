FROM python:3.10

RUN mkdir /so-vits-svc
WORKDIR /so-vits-svc

COPY . .

RUN pip install -r requirements.txt

WORKDIR /so-vits-svc/hubert

RUN curl -O https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt

EXPOSE 7860

CMD ["python", "myapp.py"]
