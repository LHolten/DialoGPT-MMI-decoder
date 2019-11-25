from asyncio import locks
from discord import Client, Message, TextChannel
from interact import append_messages, generate_message
from config import TOKEN, talk_id, view_id

client = Client()
client.message_list = []

lock = locks.Lock()


@client.event
async def on_message(person_message: Message):
    # we do not want the bot to reply to itself
    if person_message.author == client.user:
        return

    channel: TextChannel = person_message.channel
    if channel.id != talk_id:
        return

    view: TextChannel = client.get_channel(view_id)

    await lock.acquire()
    my_message = person_message.clean_content
    if my_message != '':
        await view.send(my_message)
    append_messages(client.message_list, [my_message])

    async with channel.typing():
        my_response = generate_message(client.message_list)

    if my_response != '':
        await channel.send(my_response)
        await view.send(my_response)
    append_messages(client.message_list, [my_response])
    lock.release()


@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')

client.run(TOKEN)
