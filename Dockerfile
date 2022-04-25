FROM tensorflow/tensorflow:2.7.1-gpu
COPY requirements.txt .
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
WORKDIR /current
CMD ["/bin/bash"]

#CMD ["python","DeepType.py","--data_file","/home/jupyter-pander14/fall_2021/DeepType/BRCA1View1000.mat"]
# docker run -it --rm --runtime=nvidia -v $HOME:$HOME  -v /disk:/disk -t
