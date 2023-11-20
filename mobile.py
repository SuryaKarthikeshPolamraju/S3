# Install the Twilio Python library using: pip install twilio
from twilio.rest import Client

# Your Twilio Account SID and Auth Token
account_sid = 'ACd076e0f1536c8558a0f4385842f6e4de'
auth_token = '55e5241fe17aa55dd39d4391cdddba68'

# Create a Twilio client
client = Client(account_sid, auth_token)

# Your Twilio phone number and the recipient's phone number
twilio_phone_number = '+16414274501'
recipient_phone_number = '+916309549744'  # Replace with the recipient's Indian phone number

# The message you want to send
message_text = 'Hii BSDK'

# Send the SMS
message = client.messages.create(
    body=message_text,
    from_=twilio_phone_number,
    to=recipient_phone_number
)

print(f"Message sent with SID: {message.sid}")
