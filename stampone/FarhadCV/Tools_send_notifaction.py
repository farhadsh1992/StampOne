"""
@--23.06.2023--@
Author: 
INFO:
	> send notifacrtion to pushover app into your smart phone.
	> https://pushover.net
"""





# import modules
import http.client, urllib
import psutil



class Message_Sender():
    def __init__(self, token:str=None, user:str=None):
        ## 
        self.token = "avuuii36ifd935esj8wai4w37z6kof"
        self.user = "ux3id4tq4eeqaq7siwakmswyv4apmk"
    def send_message_to_my_phone(self, title:str, message:str, url:str="", priority:int=1):


        
    
        # create connection
        conn = http.client.HTTPSConnection("api.pushover.net:443")
    	# make POST request to send message
        conn.request("POST", "/1/messages.json",
    		urllib.parse.urlencode({
        	"token": self.token,
        	"user": self.user,
       	 	"title": title,
        	"message": (message),
        	"url": url,
        	"priority": priority 
     		 }), { "Content-type": "application/x-www-form-urlencoded" })

        # get response
        conn.getresponse()
        print("message was sent")
    def check_space(self, pc_name:str="SIRIN"):
        Total, use, free = check_drive()
    
        title = f"FREE SPACE IN {pc_name}"
        message = (f"Total: {round(Total)} GiB \n"+f"Used: {round(use)} GiB \n"+f"Free: {round(free)} GiB \n")
        self.send_message_to_my_phone(title, message, priority=0)
        
    def finish_programm(self, name:str, message:str=None):
        
        title = f"Programme {name} is finished"
        self.send_message_to_my_phone(title, message, priority=0)


def check_drive():

    ## load router
    hdd = psutil.disk_usage('/')

    ## measure
    Total = round(hdd.total / (2**30))
    use   = round(hdd.used / (2**30))
    free  = round(hdd.free / (2**30))

    ## print
    print (f"Total: {Total} GiB")
    print (f"Used: {use} GiB" )
    print (f"Free: {free} GiB")

    
    return Total, use, free

	