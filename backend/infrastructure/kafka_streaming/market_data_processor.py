from typing import Any
from confluent_kafka import DeserializingConsumer, KafkaException
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer
import pandas as pd

market_data_schema = '\n{\n    "type": "record",\n    "name": "MarketData",\n    "fields": [\n        {"name": "timestamp", "type": "long"},\n        {"name": "symbol", "type": "string"},\n        {"name": "bid_price", "type": "double"},\n        {"name": "ask_price", "type": "double"},\n        {"name": "bid_size", "type": "double"},\n        {"name": "ask_size", "type": "double"},\n        {"name": "volume", "type": "double"},\n        {"name": "open_interest", "type": "double"},\n    ]\n}\n'


class MarketDataPipeline:

    def __init__(self, schema_registry_url: Any) -> Any:
        self.schema_registry = SchemaRegistryClient({"url": schema_registry_url})
        self.avro_deserializer = AvroDeserializer(
            schema_registry_client=self.schema_registry, schema_str=market_data_schema
        )
        self.consumer = DeserializingConsumer(
            {
                "bootstrap.servers": "kafka1:9092,kafka2:9092",
                "group.id": "quant_ai",
                "value.deserializer": self.avro_deserializer,
                "auto.offset.reset": "earliest",
            }
        )

    def process_real_time_ticks(self) -> Any:
        self.consumer.subscribe(["market_data"])
        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())
            yield self._transform_tick(msg.value())

    def _transform_tick(self, raw_tick: Any) -> Any:
        return {
            "timestamp": pd.to_datetime(raw_tick["timestamp"], unit="ns"),
            "symbol": raw_tick["symbol"],
            "bid": raw_tick["bid_price"],
            "ask": raw_tick["ask_price"],
            "bid_size": raw_tick["bid_size"],
            "ask_size": raw_tick["ask_size"],
            "microstructure": self._calculate_micro_features(raw_tick),
        }

    def _calculate_micro_features(self, tick: Any) -> Any:
        """Calculate microstructure features from tick data"""
        spread = tick["ask_price"] - tick["bid_price"]
        mid_price = (tick["ask_price"] + tick["bid_price"]) / 2
        imbalance = (tick["bid_size"] - tick["ask_size"]) / (
            tick["bid_size"] + tick["ask_size"]
        )
        return {
            "spread": spread,
            "spread_pct": spread / mid_price,
            "order_imbalance": imbalance,
            "liquidity_score": self._calculate_liquidity_score(tick),
        }

    def _calculate_liquidity_score(self, tick: Any) -> Any:
        """Calculate liquidity score based on order book depth and spread"""
        total_size = tick["bid_size"] + tick["ask_size"]
        spread = tick["ask_price"] - tick["bid_price"]
        return total_size / 1000 / (1 + spread)
