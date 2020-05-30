

import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)

#Next, log in to the server
server.login("saicharandgp1997@gmail.com", "saicharan13")

#Send the mail
msg = "Hello! your model has great accuracy(above 90%)" 

server.sendmail("saicharandgp1997@gmail.com", "saicharandgp@gmail.com", msg)
server.quit()
