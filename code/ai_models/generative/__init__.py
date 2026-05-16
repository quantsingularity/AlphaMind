"""
Generative Models for Synthetic Market Data
--------------------------------------------
FinancialTimeSeriesGAN  -- Vanilla GAN for return-sequence generation
MarketGAN               -- AC-GAN with auxiliary regime classifier
TransformerGenerator    -- Latent-to-sequence generator network
TimeSeriesDiscriminator -- Conv1D real/fake discriminator
RegimeClassifier        -- LSTM market-regime classifier
regime_consistency_loss -- Auxiliary loss for regime coherence
"""

from ai_models.generative.discriminator import TimeSeriesDiscriminator
from ai_models.generative.gan import FinancialTimeSeriesGAN, MarketGAN
from ai_models.generative.generator import TransformerGenerator
from ai_models.generative.regime import RegimeClassifier, regime_consistency_loss

__all__ = [
    "FinancialTimeSeriesGAN",
    "MarketGAN",
    "TransformerGenerator",
    "TimeSeriesDiscriminator",
    "RegimeClassifier",
    "regime_consistency_loss",
]
