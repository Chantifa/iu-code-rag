import logging
import os
import re
from datetime import datetime
from typing import Any, Text, Dict, List, Optional
import psycopg2
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, ActiveLoop, Restarted, AllSlotsReset, SessionStarted, ActionExecuted
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.types import DomainDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidateBookingForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_booking_form"

    def validate_name(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        logger.info(f"=== VALIDATING NAME: '{slot_value}' ===")
        if not slot_value or not isinstance(slot_value, str) or len(slot_value.strip()) < 2:
            dispatcher.utter_message(text="Please provide a valid name (at least 2 characters).")
            logger.info(f"Name validation FAILED")
            return {"name": None}
        logger.info(f"Name validation SUCCESS: '{slot_value.strip()}'")
        return {"name": slot_value.strip()}

    def validate_checkin_date(
            self,
            slot_value: str,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict
    ) -> dict:
        """Validate checkin_date slot."""
        logger.info(f"=== VALIDATING CHECKIN_DATE: '{slot_value}' ===")

        if not slot_value:
            logger.info("Checkin date is None/empty")
            return {"checkin_date": None}

        # Check if the value matches YYYY-MM-DD
        if re.match(r"^\d{4}-\d{2}-\d{2}$", slot_value):
            try:
                # Validate the date
                parsed_date = datetime.strptime(slot_value, "%Y-%m-%d")

                # Check if date is not in the past
                today = datetime.now().date()
                if parsed_date.date() < today:
                    logger.warning(f"Checkin date {parsed_date.date()} is in the past")
                    dispatcher.utter_message(text="Check-in date cannot be in the past. Please provide a future date.")
                    return {"checkin_date": None}

                logger.info(f"Checkin date parsed successfully: {parsed_date}")
                return {"checkin_date": slot_value}
            except ValueError as e:
                logger.error(f"Date parsing failed: {e}")
                dispatcher.utter_message(text=f"'{slot_value}' is not a valid date. Please use YYYY-MM-DD format.")
                return {"checkin_date": None}
        else:
            logger.warning(f"Checkin date format invalid: '{slot_value}'")
            dispatcher.utter_message(text="Please provide the check-in date in YYYY-MM-DD format.")
            return {"checkin_date": None}

    def validate_checkout_date(
            self,
            slot_value: str,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict
    ) -> dict:
        """Validate checkout_date slot."""
        logger.info(f"=== VALIDATING CHECKOUT_DATE: '{slot_value}' ===")

        if not slot_value:
            logger.info("Checkout date is None/empty")
            return {"checkout_date": None}

        # Check if the value matches YYYY-MM-DD
        if re.match(r"^\d{4}-\d{2}-\d{2}$", slot_value):
            try:
                # Validate the date
                checkout_date = datetime.strptime(slot_value, "%Y-%m-%d")
                checkin_date_str = tracker.get_slot("checkin_date")

                logger.info(f"Checkout date: {checkout_date}, Checkin date str: {checkin_date_str}")

                # Ensure checkout_date is after checkin_date
                if checkin_date_str:
                    try:
                        checkin_date = datetime.strptime(checkin_date_str, "%Y-%m-%d")
                        logger.info(f"Comparing: checkout {checkout_date.date()} vs checkin {checkin_date.date()}")

                        if checkout_date.date() <= checkin_date.date():
                            logger.warning(f"Checkout {checkout_date.date()} is not after checkin {checkin_date.date()}")
                            dispatcher.utter_message(
                                text=f"Checkout date must be after check-in date ({checkin_date_str}). Please provide a valid checkout date."
                            )
                            return {"checkout_date": None}
                    except ValueError as e:
                        logger.error(f"Error parsing checkin_date: {e}")
                        return {"checkout_date": None}

                logger.info(f"Checkout date validated successfully: {slot_value}")
                return {"checkout_date": slot_value}

            except ValueError as e:
                logger.error(f"Date parsing failed: {e}")
                dispatcher.utter_message(text=f"'{slot_value}' is not a valid date. Please use YYYY-MM-DD format.")
                return {"checkout_date": None}
        else:
            logger.warning(f"Checkout date format invalid: '{slot_value}'")
            dispatcher.utter_message(text="Please provide the checkout date in YYYY-MM-DD format.")
            return {"checkout_date": None}

    def validate_email(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        logger.info(f"=== VALIDATING EMAIL: '{slot_value}' ===")
        email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

        if not slot_value or slot_value.lower() == "none":
            logger.info("Email is None or 'none', skipping")
            return {"email": None}

        if re.match(email_regex, slot_value):
            logger.info(f"Email validated successfully: {slot_value}")
            return {"email": slot_value}
        else:
            logger.warning(f"Email validation failed: {slot_value}")
            dispatcher.utter_message(text="Please provide a valid email address (e.g., john.doe@example.com) or 'none'.")
            return {"email": None}

    def validate_phone(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        logger.info(f"=== VALIDATING PHONE: '{slot_value}' ===")
        phone_regex = r"(\+\d{1,3}[-.\s]?\d{3,}[-.\s]?\d{3,}|\d{10,})"

        if not slot_value or slot_value.lower() == "none":
            email = tracker.get_slot("email")
            if not email or email.lower() == "none":
                logger.warning("Both email and phone are None")
                dispatcher.utter_message(text="Please provide either a valid email or phone number.")
                return {"phone": None}
            logger.info("Phone is None but email is provided, allowing skip")
            return {"phone": None}

        if re.match(phone_regex, slot_value):
            logger.info(f"Phone validated successfully: {slot_value}")
            return {"phone": slot_value}

        logger.warning(f"Phone validation failed: {slot_value}")
        dispatcher.utter_message(text="Please provide a valid phone number or 'none'.")
        return {"phone": None}

    def validate_num_guests(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        logger.info(f"=== VALIDATING NUM_GUESTS: '{slot_value}' ===")
        try:
            num_guests = int(slot_value)
            if num_guests < 1:
                dispatcher.utter_message(text="Number of guests must be at least 1.")
                return {"num_guests": None}
            if num_guests > 10:
                dispatcher.utter_message(text="Maximum 10 guests allowed.")
                return {"num_guests": None}
            logger.info(f"Num guests validated: {num_guests}")
            return {"num_guests": num_guests}
        except (ValueError, TypeError):
            dispatcher.utter_message(text="Please provide a valid number of guests.")
            return {"num_guests": None}

    def validate_room_type(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        logger.info(f"=== VALIDATING ROOM_TYPE: '{slot_value}' ===")
        valid_room_types = ["single", "double", "triple", "quad", "family"]

        if slot_value and slot_value.lower() in valid_room_types:
            num_guests = tracker.get_slot("num_guests")
            if num_guests:
                room_capacities = {"single": 1, "double": 2, "triple": 3, "quad": 4, "family": 6}
                capacity = room_capacities.get(slot_value.lower(), 6)
                if int(num_guests) > capacity:
                    dispatcher.utter_message(
                        text=f"The {slot_value} room can only accommodate up to {capacity} guests, but you have {num_guests} guests.")
                    return {"room_type": None}
            logger.info(f"Room type validated: {slot_value.lower()}")
            return {"room_type": slot_value.lower()}

        dispatcher.utter_message(text="Please choose a valid room type: single, double, triple, quad, or family.")
        return {"room_type": None}

    def validate_breakfast(
        self,
        slot_value: str,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: dict
    ) -> dict:
        """Validate breakfast slot."""
        logger.info(f"=== VALIDATING BREAKFAST: '{slot_value}' ===")

        if not slot_value:
            return {"breakfast": None}

        slot_value = slot_value.lower()
        if slot_value in ["yes", "no"]:
            logger.info(f"Breakfast validated: {slot_value}")
            return {"breakfast": slot_value}
        else:
            dispatcher.utter_message(text="Please answer with 'yes' or 'no' for breakfast inclusion.")
            return {"breakfast": None}

    def validate_transport_type(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        logger.info(f"=== VALIDATING TRANSPORT_TYPE: '{slot_value}' ===")
        valid_transport_types = ["pickup", "dropoff", "both", "none"]

        if slot_value and slot_value.lower() in valid_transport_types:
            logger.info(f"Transport type validated: {slot_value.lower()}")
            return {"transport_type": slot_value.lower()}

        dispatcher.utter_message(text="Please specify a valid transport type: pickup, dropoff, both, or none.")
        return {"transport_type": None}

    def validate_transport_time_pickup(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict
    ) -> Dict[Text, Any]:
        logger.info(f"=== VALIDATING TRANSPORT_TIME_PICKUP: '{slot_value}' ===")
        logger.info(f"Type of slot_value: {type(slot_value)}")
        logger.info(f"Current transport_type: {tracker.get_slot('transport_type')}")
        logger.info(f"Latest message: {tracker.latest_message}")

        transport_type = tracker.get_slot("transport_type")

        # Skip validation if transport type doesn't require pickup
        if not transport_type or transport_type.lower() in ["none", "dropoff"]:
            logger.info("Skipping pickup time (not required for this transport type)")
            return {"transport_time_pickup": "not_needed"}

        # If no value provided, ask for it
        if not slot_value:
            logger.info("No pickup time provided - returning None")
            return {"transport_time_pickup": None}

        # Convert to string if needed
        slot_value_str = str(slot_value).strip()
        logger.info(f"Processing slot value as string: '{slot_value_str}'")

        # Validate time format
        try:
            datetime.strptime(slot_value_str, "%H:%M")
            logger.info(f"✓ Pickup time validated successfully: {slot_value_str}")
            return {"transport_time_pickup": slot_value_str}
        except ValueError as e:
            logger.warning(f"✗ Invalid time format: {slot_value_str} - Error: {e}")
            dispatcher.utter_message(text="Please provide a valid pickup time in HH:MM format (e.g., 14:00).")
            return {"transport_time_pickup": None}

    def validate_transport_time_dropoff(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        logger.info(f"=== VALIDATING TRANSPORT_TIME_DROPOFF: '{slot_value}' ===")
        transport_type = tracker.get_slot("transport_type")

        # Skip validation if transport type doesn't require dropoff
        if not transport_type or transport_type.lower() in ["none", "pickup"]:
            logger.info("Skipping dropoff time (not required for this transport type)")
            return {"transport_time_dropoff": "not_needed"}

        # If no value provided, ask for it
        if not slot_value:
            logger.info("No dropoff time provided")
            return {"transport_time_dropoff": None}

        # Validate time format
        try:
            datetime.strptime(slot_value, "%H:%M")
            logger.info(f"Dropoff time validated: {slot_value}")
            return {"transport_time_dropoff": slot_value}
        except ValueError:
            logger.warning(f"Invalid time format: {slot_value}")
            dispatcher.utter_message(text="Please provide a valid dropoff time in HH:MM format (e.g., 14:00).")
            return {"transport_time_dropoff": None}

    def validate_requirement(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        logger.info(f"=== VALIDATING REQUIREMENT: '{slot_value}' ===")
        valid_requirements = ["non-smoking", "smoking", "rolling chair", "car parking", "none"]

        if slot_value and slot_value.lower() in valid_requirements:
            logger.info(f"Requirement validated: {slot_value.lower()}")
            return {"requirement": slot_value.lower()}

        dispatcher.utter_message(text="Please specify a valid requirement: non-smoking, smoking, rolling chair, car parking, or none.")
        return {"requirement": None}

    def validate_payment_method(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict
    ) -> Dict[Text, Any]:
        logger.info(f"=== VALIDATING PAYMENT_METHOD: '{slot_value}' ===")
        valid_payment_methods = ["credit card", "pay on arrival", "debit card", "bank transfer", "paypal"]

        if slot_value and slot_value.lower() in valid_payment_methods:
            logger.info(f"Payment method validated: {slot_value.lower()}")
            return {"payment_method": slot_value.lower()}

        dispatcher.utter_message(
            text="Please choose a valid payment method: Credit Card, Pay on Arrival, Debit Card, Bank Transfer, or PayPal.")
        return {"payment_method": None}


class ActionSessionStart(Action):
    """Custom session start action to reset all slots."""

    def name(self) -> Text:
        return "action_session_start"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        logger.info("=== SESSION START - Resetting all slots ===")

        # Get all slot names from domain
        events = [SessionStarted()]

        # Reset all slots
        for slot in tracker.slots.keys():
            events.append(SlotSet(slot, None))

        # Add action_listen to continue
        events.append(ActionExecuted("action_listen"))

        return events


class ActionSaveBooking(Action):
    def name(self) -> Text:
        return "action_save_booking"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        name = tracker.get_slot("name")
        email = tracker.get_slot("email")
        phone = tracker.get_slot("phone")
        checkin_date = tracker.get_slot("checkin_date")
        checkout_date = tracker.get_slot("checkout_date")
        num_guests = tracker.get_slot("num_guests")
        room_type = tracker.get_slot("room_type")
        breakfast = tracker.get_slot("breakfast")
        transport_type = tracker.get_slot("transport_type")
        transport_time_pickup = tracker.get_slot("transport_time_pickup")
        transport_time_dropoff = tracker.get_slot("transport_time_dropoff")
        requirement = tracker.get_slot("requirement")
        payment_method = tracker.get_slot("payment_method")

        # Clean up "not_needed" values
        if transport_time_pickup == "not_needed":
            transport_time_pickup = None
        if transport_time_dropoff == "not_needed":
            transport_time_dropoff = None

        logger.info(f"=== SAVING BOOKING ===")
        logger.info(f"All slots: name={name}, email={email}, phone={phone}, checkin={checkin_date}, "
                   f"checkout={checkout_date}, guests={num_guests}, room={room_type}, breakfast={breakfast}, "
                   f"payment={payment_method}, transport={transport_type}, pickup_time={transport_time_pickup}, "
                   f"dropoff_time={transport_time_dropoff}")

        required_slots = [name, checkin_date, checkout_date, num_guests, breakfast, payment_method, room_type]
        if not all(required_slots):
            missing_slots = [slot for slot, value in [
                ("name", name), ("checkin_date", checkin_date), ("checkout_date", checkout_date),
                ("num_guests", num_guests), ("breakfast", breakfast), ("payment_method", payment_method),
                ("room_type", room_type)
            ] if not value]
            dispatcher.utter_message(
                text=f"Sorry, some required booking details are missing: {', '.join(missing_slots)}.")
            return []

        if (not email or email.lower() == "none") and (not phone or phone.lower() == "none"):
            dispatcher.utter_message(text="Please provide either an email address or phone number for contact.")
            return []

        try:
            guests_int = int(num_guests)
        except (ValueError, TypeError):
            dispatcher.utter_message(text="Invalid number of guests provided.")
            return []

        try:
            db_host = os.getenv('DB_HOST', 'postgres')
            db_port = os.getenv('DB_PORT', '5432')
            db_name = os.getenv('DB_NAME', 'hotel_bookings')
            db_user = os.getenv('DB_USER', 'postgres')
            db_password = os.getenv('DB_PASSWORD', 'postgres')

            conn = psycopg2.connect(
                dbname=db_name, user=db_user, password=db_password,
                host=db_host, port=db_port
            )
            cursor = conn.cursor()

            cursor.execute("""
                SELECT room_id, capacity FROM rooms
                WHERE room_type = %s AND total_count > (
                    SELECT COUNT(*) FROM bookings
                    WHERE room_type = %s AND check_in <= %s AND check_out >= %s
                )
                LIMIT 1
            """, (room_type, room_type, checkout_date, checkin_date))
            available_room = cursor.fetchone()

            if not available_room:
                dispatcher.utter_message(
                    text=f"Sorry, no {room_type} rooms are available for {checkin_date} to {checkout_date}.")
                cursor.close()
                conn.close()
                return []

            room_id = available_room[0]

            cursor.execute(
                """
                INSERT INTO bookings (
                    name, check_in, check_out, guests, breakfast, payment_method,
                    transport_type, transport_time_pickup, transport_time_dropoff, requirement,
                    room_type, email, phone, created_at, room_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (name, checkin_date, checkout_date, guests_int, breakfast, payment_method,
                 transport_type, transport_time_pickup, transport_time_dropoff, requirement,
                 room_type, email, phone, datetime.now(), room_id)
            )
            conn.commit()
            cursor.close()
            conn.close()

            contact_info = ", ".join(filter(None, [
                f"Email: {email}" if email and email.lower() != "none" else None,
                f"Phone: {phone}" if phone and phone.lower() != "none" else None
            ]))

            # Build transport info message
            transport_info = ""
            if transport_type and transport_type.lower() not in ["none"]:
                if transport_type.lower() == "pickup" and transport_time_pickup:
                    transport_info = f" Airport pickup scheduled at {transport_time_pickup}."
                elif transport_type.lower() == "dropoff" and transport_time_dropoff:
                    transport_info = f" Airport dropoff scheduled at {transport_time_dropoff}."
                elif transport_type.lower() == "both":
                    times = []
                    if transport_time_pickup:
                        times.append(f"pickup at {transport_time_pickup}")
                    if transport_time_dropoff:
                        times.append(f"dropoff at {transport_time_dropoff}")
                    if times:
                        transport_info = f" Airport transport: {' and '.join(times)}."

            dispatcher.utter_message(
                text=f"Thank you, {name}! Your booking is confirmed for {num_guests} guests in a {room_type} room "
                     f"from {checkin_date} to {checkout_date}.{transport_info} Contact: {contact_info or 'N/A'}")

            logger.info("Booking saved successfully!")
        except Exception as e:
            logger.error(f"Failed to save booking: {str(e)}")
            dispatcher.utter_message(text=f"Failed to save booking: {str(e)}")
            return []

        # Reset all slots after successful booking
        events = []
        for slot in tracker.slots.keys():
            events.append(SlotSet(slot, None))
        events.append(ActiveLoop(None))

        return events


class ActionRestart(Action):
    """Restart the conversation and clear all slots."""

    def name(self) -> Text:
        return "action_restart"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        logger.info("=== RESTARTING CONVERSATION ===")
        return [Restarted(), AllSlotsReset()]