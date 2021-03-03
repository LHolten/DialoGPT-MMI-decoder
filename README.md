# DialoGPT-MMI-decoder
This is a third party implementation of the MMI-decoder for [DialoGPT](https://github.com/microsoft/DialoGPT).

It gives interesting responses, varying between funny, stupid and thoughtfull.

This is how the bot describes itself:
> Not sure how to describe it. But the person I was speaking to said it was like a little book of secrets.

### Features
* Unlimitted chat length
* Discord bot
* Discord bot tested by a group of ~15 people

## Installation

#### Python requirements:
* pytorch
* transformers

#### Other files
* Download the medium forward and backward pre-trained model from [here](https://github.com/microsoft/DialoGPT#models)
* Download the config files from [here](https://github.com/microsoft/DialoGPT/tree/master/configs)

Place all of these files in the `/medium` folder

#### Python requirement for the discord bot:
* discord.py

## Configuration
Modify `config.py`

## Usage
Run `interact.py` to chat.

Run `discord_bot.py` to run the discord bot
