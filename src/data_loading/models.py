"""
Data models for the logistics insights engine.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List


@dataclass
class Client:
    """Client data model."""
    client_id: int
    client_name: str
    gst_number: str
    contact_person: str
    contact_phone: str
    contact_email: str
    address_line1: str
    address_line2: Optional[str]
    city: str
    state: str
    pincode: str
    created_at: datetime


@dataclass
class Driver:
    """Driver data model."""
    driver_id: int
    driver_name: str
    phone: str
    license_number: str
    partner_company: str
    city: str
    state: str
    status: str
    created_at: datetime


@dataclass
class ExternalFactor:
    """External factor data model."""
    factor_id: int
    order_id: int
    traffic_condition: str
    weather_condition: str
    event_type: Optional[str]
    recorded_at: datetime


@dataclass
class Feedback:
    """Customer feedback data model."""
    feedback_id: int
    order_id: int
    customer_name: str
    feedback_text: str
    sentiment: str
    rating: int
    created_at: datetime


@dataclass
class FleetLog:
    """Fleet log data model."""
    fleet_log_id: int
    order_id: int
    driver_id: int
    vehicle_number: str
    route_code: str
    gps_delay_notes: Optional[str]
    departure_time: datetime
    arrival_time: datetime
    created_at: datetime


@dataclass
class Order:
    """Order data model."""
    order_id: int
    client_id: int
    customer_name: str
    customer_phone: str
    delivery_address_line1: str
    delivery_address_line2: Optional[str]
    city: str
    state: str
    pincode: str
    order_date: datetime
    promised_delivery_date: datetime
    actual_delivery_date: Optional[datetime]
    status: str
    payment_mode: str
    amount: float
    failure_reason: Optional[str]
    created_at: datetime


@dataclass
class WarehouseLog:
    """Warehouse log data model."""
    log_id: int
    order_id: int
    warehouse_id: int
    picking_start: datetime
    picking_end: datetime
    dispatch_time: datetime
    notes: Optional[str]


@dataclass
class Warehouse:
    """Warehouse data model."""
    warehouse_id: int
    warehouse_name: str
    state: str
    city: str
    pincode: str
    capacity: int
    manager_name: str
    contact_phone: str
    created_at: datetime
