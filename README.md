# MlOps
Firstly we make things ready for building the MlOps pipeline.

 Create a container image that has Python3 and Keras or NumPy installed using a docker file
When we launch this image, it should automatically start to train the model in the container.

Create a job chain of job1, job2, job3, job4, job5 using build Pipeline plugin in Jenkins.    

 secondly, we go by performing each task.



Job1: 

Pull the Github repo automatically when some developers push the repo to Github.



Job2 : 

By looking at the code or program file, Jenkins should automatically start the respective machine learning software installed interpreter install image container to deploy code and start training( eg. If code uses CNN, then Jenkins should start the container that has already installed all the software required for the CNN processing).



Job3 : 

Train your model and predict accuracy or metrics.



Job4 :

 if metrics accuracy is less than 90%, then tweak the machine learning model architecture.



Job5: 

Retrain the model or notify that the best model is being created.



for the reference of the code, I have created  the GitHub repo.
