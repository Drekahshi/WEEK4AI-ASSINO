# AgriMedFinance AI with Kenya Finance Bill Gas Excise Duty Implementation
# Demonstrates tiered pricing based on gas consumption similar to excise duty principles

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import hashlib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class KenyaGasExciseDuty:
    """
    Implementation of Kenya Finance Bill-inspired excise duty on smart contract gas usage.
    Based on tiered pricing structure similar to excise duty on various goods and services.
    """

    def __init__(self):
        # Gas tiers based on Kenya Finance Bill excise duty principles
        self.gas_tiers = {
            'basic_tier': {
                'threshold': 30000,  # Free tier up to 30,000 gas units
                'rate': 0.0,         # 0% - Free basic services
                'description': 'Basic Service Tier (Free)'
            },
            'standard_tier': {
                'threshold': 75000,  # Standard tier 30,001 - 75,000 gas units
                'rate': 0.05,        # 5% excise duty
                'description': 'Standard Service Tier (5% Excise)'
            },
            'premium_tier': {
                'threshold': 150000, # Premium tier 75,001 - 150,000 gas units
                'rate': 0.12,        # 12% excise duty
                'description': 'Premium Service Tier (12% Excise)'
            },
            'luxury_tier': {
                'threshold': float('inf'), # Luxury tier 150,001+ gas units
                'rate': 0.20,        # 20% excise duty (luxury goods rate)
                'description': 'Luxury Service Tier (20% Excise)'
            }
        }

        # Base gas price per unit (in KES)
        self.base_gas_price = 0.001  # 0.001 KES per gas unit

        # VAT rate (as per Kenya Finance Bill)
        self.vat_rate = 0.16  # 16% VAT

        self.transaction_log = []

    def calculate_gas_cost(self, gas_used, service_type="general"):
        """
        Calculate the total cost including excise duty based on gas consumption.
        Implements progressive taxation similar to Kenya's excise duty structure.
        """

        base_cost = gas_used * self.base_gas_price
        excise_duty = 0
        tier_applied = ""

        # Determine excise duty tier
        if gas_used <= self.gas_tiers['basic_tier']['threshold']:
            excise_rate = self.gas_tiers['basic_tier']['rate']
            tier_applied = "Basic (Free)"
        elif gas_used <= self.gas_tiers['standard_tier']['threshold']:
            # Free tier + excise on excess
            excess_gas = gas_used - self.gas_tiers['basic_tier']['threshold']
            excise_duty = excess_gas * self.base_gas_price * self.gas_tiers['standard_tier']['rate']
            tier_applied = "Standard (5%)"
        elif gas_used <= self.gas_tiers['premium_tier']['threshold']:
            # Calculate tiered excise
            standard_excess = self.gas_tiers['standard_tier']['threshold'] - self.gas_tiers['basic_tier']['threshold']
            premium_excess = gas_used - self.gas_tiers['standard_tier']['threshold']

            excise_duty = (standard_excess * self.base_gas_price * self.gas_tiers['standard_tier']['rate'] +
                           premium_excess * self.base_gas_price * self.gas_tiers['premium_tier']['rate'])
            tier_applied = "Premium (12%)"
        else:
            # Luxury tier - highest excise rate
            standard_excess = self.gas_tiers['standard_tier']['threshold'] - self.gas_tiers['basic_tier']['threshold']
            premium_excess = self.gas_tiers['premium_tier']['threshold'] - self.gas_tiers['standard_tier']['threshold']
            luxury_excess = gas_used - self.gas_tiers['premium_tier']['threshold']

            excise_duty = (standard_excess * self.base_gas_price * self.gas_tiers['standard_tier']['rate'] +
                           premium_excess * self.base_gas_price * self.gas_tiers['premium_tier']['rate'] +
                           luxury_excess * self.base_gas_price * self.gas_tiers['luxury_tier']['rate'])
            tier_applied = "Luxury (20%)"

        # Calculate VAT on (base cost + excise duty)
        subtotal = base_cost + excise_duty
        vat_amount = subtotal * self.vat_rate

        total_cost = subtotal + vat_amount

        # Log transaction
        transaction_record = {
            'timestamp': datetime.now(),
            'gas_used': gas_used,
            'service_type': service_type,
            'base_cost': base_cost,
            'excise_duty': excise_duty,
            'vat_amount': vat_amount,
            'total_cost': total_cost,
            'tier_applied': tier_applied
        }

        self.transaction_log.append(transaction_record)

        return {
            'gas_used': gas_used,
            'base_cost': round(base_cost, 4),
            'excise_duty': round(excise_duty, 4),
            'vat_amount': round(vat_amount, 4),
            'total_cost': round(total_cost, 4),
            'tier_applied': tier_applied,
            'effective_rate': round((total_cost / base_cost - 1) * 100, 2) if base_cost > 0 else 0
        }

    def get_pricing_summary(self):
        """Generate a summary of the excise duty pricing structure"""
        return self.gas_tiers

    def analyze_usage_patterns(self):
        """Analyze gas usage patterns and excise duty collection"""
        if not self.transaction_log:
            return "No transactions recorded yet."

        df = pd.DataFrame(self.transaction_log)

        analysis = {
            'total_transactions': len(df),
            'total_gas_consumed': df['gas_used'].sum(),
            'total_excise_collected': df['excise_duty'].sum(),
            'total_vat_collected': df['vat_amount'].sum(),
            'average_gas_per_transaction': df['gas_used'].mean(),
            'tier_distribution': df['tier_applied'].value_counts().to_dict()
        }

        return analysis

class EnhancedSmartContract:
    """Enhanced Smart Contract with Kenya Finance Bill Gas Excise Implementation"""

    def __init__(self, contract_type):
        self.contract_type = contract_type
        self.transactions = []
        self.balance = 1000000  # Starting balance in wei
        self.gas_excise = KenyaGasExciseDuty()

    def execute_transaction(self, function_name, params, sender, estimated_gas=None):
        """Execute transaction with Kenya Finance Bill gas excise calculations"""

        # Simulate gas usage based on function complexity
        if estimated_gas:
            gas_used = estimated_gas
        else:
            base_gas = {
                'submitClaim': np.random.randint(35000, 65000),
                'createInsurancePolicy': np.random.randint(45000, 85000),
                'applyForLoan': np.random.randint(30000, 70000),
                'processPayment': np.random.randint(25000, 45000)
            }