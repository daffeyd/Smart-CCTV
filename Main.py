import telebot
import pyttsx3

TOKEN = '1671546396:AAHhcAcuOKFKqM514zg9TLDTgnhyKOJws_Q'
bot = telebot.TeleBot(TOKEN)

import time



def findat(msg):
    # from a list of texts, it finds the one with the '@' sign
    for i in msg:
        if '/' in i:
            return i

@bot.message_handler(commands=['start']) # welcome message handler
def send_welcome(message):
    bot.reply_to(message, '(placeholder text)')

@bot.message_handler(commands=['help']) # help message handler
def send_welcome(message):
    bot.reply_to(message, 'ALPHA = FEATURES MAY NOT WORK')

@bot.message_handler(func=lambda msg: msg.text is not None and '/' in msg.text)
# lambda function finds messages with the '@' sign in them
# in case msg.text doesn't exist, the handler doesn't process it
def at_converter(message):
    engine= pyttsx3.init()
    texts = message.text.split()
    at_text = findat(texts)
    if at_text == '@': # in case it's just the '@', skip
        pass
    else:
        bot.reply_to(message, 'Your message will be speak in speaker')
        separator = ' '
        Content = separator.join(texts[1:])
        print(Content)
        engine.say(Content)
    engine.runAndWait()
while True:
    bot.polling()
        # ConnectionError and ReadTimeout because of possible timout of the requests library
        # maybe there are others, therefore Exception
      
