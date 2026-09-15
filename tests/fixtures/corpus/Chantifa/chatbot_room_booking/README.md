# Hotel Booking Chatbot

A conversational AI chatbot built with Rasa for hotel room bookings, featuring comprehensive booking management, room availability checking, and integration with Telegram.

## 🌟 Features

- **Intelligent Conversation Flow**: Natural language understanding for hotel booking interactions
- **Complete Booking Management**: Handles guest information, room selection, dates, and special requirements
- **Room Availability Checking**: Real-time validation against PostgreSQL database
- **Airport Transport Coordination**: Pickup and dropoff scheduling with time management
- **Multiple Payment Methods**: Credit card, debit card, bank transfer, PayPal, and pay on arrival
- **Special Requirements**: Non-smoking rooms, accessibility features, parking, and more
- **Telegram Integration**: Connect with users via Telegram messenger
- **Data Validation**: Comprehensive validation for dates, contact info, guest count, and room capacity
- **Database Persistence**: PostgreSQL storage for bookings and conversation tracking

## 🏗️ Architecture

The system consists of four main components:

1. **Rasa Server** (Port 5005): Handles conversation management and NLU
2. **Action Server** (Port 5055): Executes custom actions and database operations
3. **PostgreSQL Database** (Port 5432): Stores bookings, rooms, and conversation events
4. **Ngrok** (Port 4040): Exposes the chatbot for Telegram webhook integration

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Docker** (version 20.x or higher)
- **Docker Compose** (version 1.29.x or higher)
- **Git** (for cloning the repository)
- **Ngrok Account** (for Telegram integration - free tier available)

### System Requirements

- **RAM**: Minimum 4GB, recommended 8GB
- **Disk Space**: At least 5GB free space
- **OS**: Linux, macOS, or Windows with WSL2

## 🚀 Installation Manual

### Step 1: Clone the Repository

```bash
git clone https://github.com/Chantifa/chatbot_room_booking.git
cd hotel-booking-chatbot
```

### Step 2: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# .env
NGROK_AUTHTOKEN=your_ngrok_auth_token_here
TELEGRAM_ACCESS_TOKEN=you_telegram_access_token
```

To get your Ngrok auth token:
1. Sign up at [ngrok.com](https://ngrok.com)
2. Go to your dashboard
3. Copy your auth token

### Step 3: Configure Telegram Bot (Optional)

If you want to use Telegram integration:

1. **Create a Telegram Bot**:
   - Open Telegram and search for `@BotFather`
   - Send `/newbot` and follow the instructions
   - Save the API token you receive

2. **Update credentials.yml**:
   ```yaml
   telegram:
     access_token: "YOUR_TELEGRAM_BOT_TOKEN"
     verify: "YourBotName"
     webhook_url: "YOUR_NGROK_URL/webhooks/telegram/webhook"
   ```

3. **Update ngrok.yml** with your ngrok domain:
   ```yaml
   version: "2"
   tunnels:
     rasa:
       addr: rasa:5005
       proto: http
       hostname: your-ngrok-domain.ngrok-free.app
   ```

### Step 4: Build and Start the Services

```bash
# Build the Docker images
docker-compose build

# Start all services
docker-compose up -d
```

This will:
- Build the Rasa and action server images
- Start PostgreSQL database and initialize tables
- Train the Rasa model (first run only)
- Start all services in detached mode

### Step 5: Verify Installation

Check that all services are running:

```bash
docker-compose ps
```

You should see all four services (rasa, action_server, postgres, ngrok) with status "Up".

### Step 6: Setup webhook

```bash
# Set up webhook for the communication between Telegram and ngrok server

curl -X GET https://api.telegram.org/bot<TELEGRAM_ACCESS_TOKEN>/setWebhook?url=https://<ngrok-endpoint-Id>.ngrok-free.app/webhooks/telegram/webhook

```
### Step 7: Test the Chatbot

#### Option A: Test via REST API

```bash
curl -X POST http://localhost:5005/webhooks/rest/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "sender": "test_user",
    "message": "hello"
  }'
```

#### Option B: Test via Rasa Shell (Interactive)

```bash
docker-compose exec rasa rasa shell
```

Type your messages and interact with the bot directly.

#### Option C: Test via Telegram

1. Get your ngrok URL:
   ```bash
   curl http://localhost:4040/api/tunnels
   ```

2. Set the webhook in Telegram:
   - Update the `webhook_url` in `credentials.yml`
   - Restart the services: `docker-compose restart rasa`

3. Find your bot on Telegram and start chatting!

## 📖 Usage Guide

### Starting a Booking Conversation

1. Greet the bot: `hello`, `hi`, `hey`
2. Follow the prompts to provide:
   - Your name
   - Check-in date (YYYY-MM-DD format)
   - Check-out date (YYYY-MM-DD format)
   - Number of guests
   - Room type (single, double, triple, quad, family)
   - Breakfast preference (yes/no)
   - Contact information (email and/or phone)
   - Airport transport needs (pickup, dropoff, both, none)
   - Special requirements (non-smoking, parking, etc.)
   - Payment method

### Room Types and Capacity

| Room Type | Capacity | Available Rooms |
|-----------|----------|-----------------|
| Single    | 1 guest  | 20              |
| Double    | 2 guests | 40              |
| Triple    | 3 guests | 20              |
| Quad      | 4 guests | 10              |
| Family    | 6 guests | 10              |

### Example Conversation

```
User: hello
Bot: Hello! Welcome to our hotel booking service. I'll help you make a reservation. Let's start with your name.

User: John Doe
Bot: What's your check-in date? (Please use format: YYYY-MM-DD, e.g., 2025-12-25)

User: 2025-12-20
Bot: What's your check-out date? (Please use format: YYYY-MM-DD, e.g., 2025-12-28)

User: 2025-12-23
Bot: How many guests will be staying?

User: 2
Bot: What type of room would you like? (single, double, triple, quad, or family)

User: double
Bot: Would you like breakfast included? (yes or no)

User: yes
Bot: What's your email address? (or type 'none' if you prefer not to provide it)

User: john.doe@example.com
Bot: What's your phone number? (or type 'none' if you prefer not to provide it)

User: none
Bot: Do you need airport transport? (pickup, dropoff, both, or none)

User: pickup
Bot: What time do you need pickup? (HH:MM format, e.g., 14:00)

User: 14:30
Bot: Do you have any special requirements? (non-smoking, smoking, rolling chair, car parking, or none)

User: non-smoking
Bot: How would you like to pay? (Credit Card, Pay on Arrival, Debit Card, Bank Transfer, or PayPal)

User: credit card
Bot: Thank you, John Doe! Your booking is confirmed for 2 guests in a double room from 2025-12-20 to 2025-12-23. Airport pickup scheduled at 14:30. Contact: Email: john.doe@example.com
```

## 🛠️ Management Commands

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f rasa
docker-compose logs -f action_server
docker-compose logs -f postgres
```

### Restart Services

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart rasa
```

### Retrain the Model

```bash
docker-compose exec rasa rasa train
```

### Access the Database

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d hotel_bookings

# View all bookings
SELECT * FROM bookings;

# View room availability
SELECT * FROM rooms;

# Exit database
\q
```

### Reset Conversation

While chatting, send: `restart` or `start over`

### Stop All Services

```bash
docker-compose down
```

### Stop and Remove All Data

```bash
docker-compose down -v
```

## 📁 Project Structure

```
.
├── actions.py                  # Custom actions and form validations
├── config.yml                  # NLU pipeline and policy configuration
├── credentials.yml             # Telegram and other channel credentials
├── domain.yml                  # Intents, entities, slots, and responses
├── endpoints.yml               # Action server and tracker store configuration
├── docker-compose.yml          # Multi-container Docker configuration
├── Dockerfile                  # Rasa environment setup
├── ngrok.yml                   # Ngrok tunnel configuration
├── data/
│   ├── nlu.yaml               # Training data for NLU
│   ├── rules.yaml             # Conversation rules
│   ├── stories.yaml           # Training stories
│   └── init.sql               # Database initialization script
└── models/                     # Trained models (generated)
```

## 🐛 Troubleshooting

### Issue: Services won't start

**Solution**: Check if ports are already in use
```bash
# Check port usage
netstat -an | grep 5005
netstat -an | grep 5055
netstat -an | grep 5432

# Stop conflicting services or change ports in docker-compose.yml
```

### Issue: Model training fails

**Solution**: Increase Docker memory allocation
- Docker Desktop: Settings → Resources → Memory (increase to 4GB+)

### Issue: Database connection fails

**Solution**: Wait for PostgreSQL to be ready
```bash
# Check database health
docker-compose exec postgres pg_isready -U postgres

# Restart if needed
docker-compose restart postgres
```

### Issue: Telegram webhook not working

**Solution**: 
1. Verify ngrok is running: `curl http://localhost:4040/api/tunnels`
2. Check webhook URL in credentials.yml matches ngrok URL
3. Restart rasa service: `docker-compose restart rasa`

### Issue: "No rooms available" error

**Solution**: Check room data in database
```bash
docker-compose exec postgres psql -U postgres -d hotel_bookings -c "SELECT * FROM rooms;"
```

If rooms table is empty, reinitialize:
```bash
docker-compose down -v
docker-compose up -d
```

## 🔒 Security Notes

- **Never commit** sensitive credentials (tokens, passwords) to version control
- Use environment variables for all secrets
- Rotate Telegram bot tokens periodically
- Use strong passwords for database in production
- Enable SSL/TLS for production deployments

## 📝 Development

### Adding New Intents

1. Add training examples to `data/nlu.yaml`
2. Add intent to `domain.yml`
3. Add handling rules/stories to `data/rules.yaml` or `data/stories.yaml`
4. Retrain: `docker-compose exec rasa rasa train`

### Modifying Form Validation

Edit the validation methods in `actions.py` under the `ValidateBookingForm` class.

### Database Schema Changes

1. Modify `data/init.sql`
2. Recreate database: `docker-compose down -v && docker-compose up -d`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 📧 Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review Rasa documentation: https://rasa.com/docs/

## 🙏 Acknowledgments

- Built with [Rasa Open Source](https://rasa.com/)
- Database: [PostgreSQL](https://www.postgresql.org/)
- Containerization: [Docker](https://www.docker.com/)
- Tunneling: [Ngrok](https://ngrok.com/)

---

**Version**: 1.0.0  
**Last Updated**: November 2025
