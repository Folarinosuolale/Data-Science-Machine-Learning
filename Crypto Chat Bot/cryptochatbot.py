#THIS PROJECT IS FOCUSED ON CREATING CRYPTOCURRENCY CHATBOT

from nltk.chat.util import Chat, reflections

#Pairs is a list of patterns and responses.
pairs = [
    [
        r"(.*)my name is (.*)",
        ["Hello %2, How are you today ?",]
    ],
    [
        r"(.*)help(.*) ",
        ["I can help you ",]
    ],
     [
        r"(.*) your name ?",
        ["I don't have a name but I'm a crypto chat bot..",]
    ],
    [
        r"how are you (.*) ?",
        ["I'm doing very well, thank you!", "I am great, thank you!"]
    ],
    [
        r"sorry (.*)",
        ["Its alright","Its OK, never mind that",]
    ],
    [
        r"I'm (.*) (good|well|okay|ok|fine)",
        ["Nice to hear that","Alright, great !",]
    ],
    [
        r"(hi|hey|hello|hola|holla)(.*)",
        ["Hello. I'm TheCryptoBot, How may I assist you?", "Hey there. I'm TheCryptoBot, How may I assist you?",]
    ],
    [
        r"Give me examples of Cryptocurrencies?",
        ["Bitcoin, Ethereum, Tether, Litecoin, Solana, XRP, Cardano, etc.",]
        
    ],
    [
        r"(.*)created(.*)",
        ["Folarin Osuolale created me using Python's NLTK library ","top secret ;)",]
    ],
    [
        r"(.*) (location|city) ?",
        ['I work anywhere. But I was created in Lagos, Nigeria',]
    ],
    [
        r"what is cryptocurrency?",
        ["A cryptocurrency is a digital currency, which is an alternative form of payment created using encryption algorithms",]
    ],
    [
        r"how (.*) health (.*)",
        ["Health is very important, but I am a computer, so I don't need to worry about my health ",]
    ],
    [
        r"what is cryptocurrency built on?",
        ["Cryptocurrencies are usually built using blockchain technology. Blockchain describes the way transactions are recorded into 'blocks' and time stamped. It's a fairly complex, technical process, but the result is a digital ledger of cryptocurrency transactions that's hard for hackers to tamper with",]
    ],
    [
        r"Can I make money from crypto?",
        ["Yes, You can make money, and fast, with cryptocurrency. However, you must follow industry news, know crypto trading best practices, guard against theft, do your research on coins and crypto exchanges and have an exit strategy ready in case you need one."]
    ],
    [
        r"quit",
        ["Bye for now. See you soon :) ","It was nice talking to you. See you soon :)"]
    ],
    [
        r"(.*)",
        ["Ask me what you'd like to know",]
    ],
]

#default message at the start of chat
print("Hi, I'm TheCryptoBot and I like to chat about cryptocurrency\nPlease type lowercase English language to start a conversation. Type quit to leave ")

#Creating the Chat Bot
chat = Chat(pairs, reflections)

#Starting a conversation
chat.converse()