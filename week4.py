# AgriMedFinance AI Ecosystem - Python Demo
# Run this in Google Colab or Jupyter Notebook

# Install required packages (run this first in Colab)
# !pip install scikit-learn pandas numpy matplotlib seaborn

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

class SmartContract:
    """Simulated Smart Contract for demonstration"""
    
    def __init__(self, contract_type):
        self.contract_type = contract_type
        self.transactions = []
        self.balance = 1000000  # Starting balance in wei
        
    def execute_transaction(self, function_name, params, sender):
        """Simulate smart contract execution"""
        gas_used = np.random.randint(21000, 100000)
        success = True
        
        transaction = {
            'tx_hash': hashlib.sha256(f"{function_name}{datetime.now()}".encode()).hexdigest()[:16],
            'function': function_name,
            'params': params,
            'sender': sender,
            'gas_used': gas_used,
            'timestamp': datetime.now(),
            'success': success
        }
        
        self.transactions.append(transaction)
        return transaction

class TelemedicineAI:
    """AI Agent for Telemedicine Module"""
    
    def __init__(self):
        self.diagnostic_model = self._create_diagnostic_model()
        self.treatment_model = self._create_treatment_model()
        self.contract = SmartContract("Medical_Insurance")
        
    def _create_diagnostic_model(self):
        """Create a simple diagnostic model"""
        # Simulate training data
        np.random.seed(42)
        n_samples = 1000
        
        # Features: age, symptoms_count, vital_signs, lab_results
        X = np.random.randn(n_samples, 4)
        X[:, 0] = np.random.randint(18, 80, n_samples)  # age
        X[:, 1] = np.random.randint(1, 10, n_samples)  # symptoms count
        
        # Target: 0=healthy, 1=needs_consultation, 2=emergency
        y = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return model
    
    def _create_treatment_model(self):
        """Create treatment recommendation model"""
        np.random.seed(42)
        n_samples = 500
        
        # Features: diagnosis_severity, patient_age, medical_history
        X = np.random.randn(n_samples, 3)
        # Target: treatment_cost
        y = np.random.uniform(50, 2000, n_samples)
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return model
    
    def diagnose_patient(self, patient_data):
        """AI-powered patient diagnosis"""
        features = np.array([
            patient_data['age'],
            patient_data['symptoms_count'],
            patient_data['blood_pressure'],
            patient_data['lab_results']
        ]).reshape(1, -1)
        
        diagnosis = self.diagnostic_model.predict(features)[0]
        confidence = max(self.diagnostic_model.predict_proba(features)[0])
        
        diagnosis_map = {0: "Healthy", 1: "Needs Consultation", 2: "Emergency"}
        
        return {
            'diagnosis': diagnosis_map[diagnosis],
            'confidence': confidence,
            'recommendation': self._get_recommendation(diagnosis)
        }
    
    def _get_recommendation(self, diagnosis):
        recommendations = {
            0: "Continue regular health monitoring",
            1: "Schedule consultation within 48 hours",
            2: "Immediate medical attention required"
        }
        return recommendations[diagnosis]
    
    def process_insurance_claim(self, patient_id, diagnosis, estimated_cost):
        """Process insurance claim through smart contract"""
        if diagnosis in ["Needs Consultation", "Emergency"]:
            # Auto-approve claims under $500 or emergency cases
            auto_approve = estimated_cost < 500 or diagnosis == "Emergency"
            
            claim_data = {
                'patient_id': patient_id,
                'diagnosis': diagnosis,
                'amount': estimated_cost,
                'auto_approved': auto_approve
            }
            
            tx = self.contract.execute_transaction(
                'submitClaim', 
                claim_data, 
                f"patient_{patient_id}"
            )
            
            return {
                'claim_approved': auto_approve,
                'transaction_hash': tx['tx_hash'],
                'gas_used': tx['gas_used']
            }
        
        return {'claim_approved': False, 'reason': 'No treatment needed'}

class AgricultureAI:
    """AI Agent for Agriculture Module"""
    
    def __init__(self):
        self.crop_health_model = self._create_crop_model()
        self.yield_prediction_model = self._create_yield_model()
        self.weather_model = self._create_weather_model()
        self.contract = SmartContract("Crop_Insurance")
        
    def _create_crop_model(self):
        """Create crop health assessment model"""
        np.random.seed(42)
        n_samples = 800
        
        # Features: soil_moisture, temperature, rainfall, fertilizer_used
        X = np.random.randn(n_samples, 4)
        X[:, 0] = np.random.uniform(0.2, 0.8, n_samples)  # soil moisture
        X[:, 1] = np.random.uniform(15, 35, n_samples)    # temperature
        X[:, 2] = np.random.uniform(0, 200, n_samples)    # rainfall
        X[:, 3] = np.random.uniform(0, 100, n_samples)    # fertilizer
        
        # Target: crop health score (0-100)
        y = np.random.uniform(40, 100, n_samples)
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return model
    
    def _create_yield_model(self):
        """Create crop yield prediction model"""
        np.random.seed(42)
        n_samples = 600
        
        X = np.random.randn(n_samples, 5)
        y = np.random.uniform(2, 8, n_samples)  # tons per hectare
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return model
    
    def _create_weather_model(self):
        """Create weather prediction model"""
        np.random.seed(42)
        n_samples = 400
        
        X = np.random.randn(n_samples, 3)
        y = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 0=normal, 1=extreme weather
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return model
    
    def assess_crop_health(self, field_data):
        """AI-powered crop health assessment"""
        features = np.array([
            field_data['soil_moisture'],
            field_data['temperature'],
            field_data['rainfall'],
            field_data['fertilizer_used']
        ]).reshape(1, -1)
        
        health_score = self.crop_health_model.predict(features)[0]
        
        return {
            'health_score': health_score,
            'status': 'Good' if health_score > 70 else 'Needs Attention' if health_score > 50 else 'Poor',
            'recommendation': self._get_crop_recommendation(health_score)
        }
    
    def predict_yield(self, crop_data):
        """Predict crop yield"""
        features = np.random.randn(1, 5)  # Simplified for demo
        yield_prediction = self.yield_prediction_model.predict(features)[0]
        
        return {
            'predicted_yield': round(yield_prediction, 2),
            'confidence_interval': (round(yield_prediction * 0.9, 2), round(yield_prediction * 1.1, 2))
        }
    
    def _get_crop_recommendation(self, health_score):
        if health_score > 80:
            return "Maintain current practices"
        elif health_score > 60:
            return "Increase irrigation and monitor closely"
        else:
            return "Immediate intervention required - check soil and pest conditions"
    
    def process_crop_insurance(self, farmer_id, crop_type, coverage_amount):
        """Process crop insurance through smart contract"""
        premium = coverage_amount * 0.05  # 5% premium rate
        
        insurance_data = {
            'farmer_id': farmer_id,
            'crop_type': crop_type,
            'coverage': coverage_amount,
            'premium': premium
        }
        
        tx = self.contract.execute_transaction(
            'createInsurancePolicy',
            insurance_data,
            f"farmer_{farmer_id}"
        )
        
        return {
            'policy_created': True,
            'premium': premium,
            'transaction_hash': tx['tx_hash']
        }

class FinanceAI:
    """AI Agent for Finance Module"""
    
    def __init__(self):
        self.credit_model = self._create_credit_model()
        self.fraud_model = self._create_fraud_model()
        self.risk_model = self._create_risk_model()
        self.contract = SmartContract("Smart_Lending")
        
    def _create_credit_model(self):
        """Create credit scoring model"""
        np.random.seed(42)
        n_samples = 1000
        
        # Features: income, debt_ratio, payment_history, assets
        X = np.random.randn(n_samples, 4)
        X[:, 0] = np.random.uniform(20000, 150000, n_samples)  # income
        X[:, 1] = np.random.uniform(0.1, 0.8, n_samples)      # debt ratio
        X[:, 2] = np.random.uniform(0.5, 1.0, n_samples)      # payment history
        X[:, 3] = np.random.uniform(10000, 500000, n_samples) # assets
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Target: credit score (300-850)
        y = np.random.uniform(300, 850, n_samples)
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_scaled, y)
        
        return {'model': model, 'scaler': scaler}
    
    def _create_fraud_model(self):
        """Create fraud detection model"""
        np.random.seed(42)
        n_samples = 800
        
        X = np.random.randn(n_samples, 6)
        y = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # 5% fraud rate
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return model
    
    def _create_risk_model(self):
        """Create risk assessment model"""
        np.random.seed(42)
        n_samples = 500
        
        X = np.random.randn(n_samples, 4)
        y = np.random.uniform(0.01, 0.15, n_samples)  # Risk score 1-15%
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return model
    
    def calculate_credit_score(self, applicant_data):
        """AI-powered credit scoring"""
        features = np.array([
            applicant_data['income'],
            applicant_data['debt_ratio'],
            applicant_data['payment_history'],
            applicant_data['assets']
        ]).reshape(1, -1)
        
        features_scaled = self.credit_model['scaler'].transform(features)
        credit_score = self.credit_model['model'].predict(features_scaled)[0]
        
        return {
            'credit_score': int(credit_score),
            'rating': self._get_credit_rating(credit_score),
            'approval_probability': self._get_approval_probability(credit_score)
        }
    
    def detect_fraud(self, transaction_data):
        """AI-powered fraud detection"""
        features = np.random.randn(1, 6)  # Simplified for demo
        fraud_probability = self.fraud_model.predict_proba(features)[0][1]
        
        return {
            'is_fraud': fraud_probability > 0.5,
            'fraud_probability': fraud_probability,
            'risk_level': 'High' if fraud_probability > 0.7 else 'Medium' if fraud_probability > 0.3 else 'Low'
        }
    
    def _get_credit_rating(self, score):
        if score >= 750:
            return "Excellent"
        elif score >= 700:
            return "Good"
        elif score >= 650:
            return "Fair"
        else:
            return "Poor"
    
    def _get_approval_probability(self, score):
        return min(max((score - 300) / 550, 0), 1)
    
    def process_loan_application(self, applicant_id, loan_amount, collateral):
        """Process loan through smart contract"""
        credit_info = self.calculate_credit_score({
            'income': np.random.uniform(30000, 100000),
            'debt_ratio': np.random.uniform(0.2, 0.6),
            'payment_history': np.random.uniform(0.7, 1.0),
            'assets': np.random.uniform(50000, 300000)
        })
        
        interest_rate = max(0.05, 0.15 - (credit_info['credit_score'] - 300) / 5500)
        approved = credit_info['credit_score'] > 650
        
        loan_data = {
            'applicant_id': applicant_id,
            'amount': loan_amount,
            'interest_rate': interest_rate,
            'collateral': collateral,
            'approved': approved,
            'credit_score': credit_info['credit_score']
        }
        
        tx = self.contract.execute_transaction(
            'applyForLoan',
            loan_data,
            f"applicant_{applicant_id}"
        )
        
        return {
            'loan_approved': approved,
            'interest_rate': interest_rate,
            'credit_score': credit_info['credit_score'],
            'transaction_hash': tx['tx_hash']
        }

class AgenticOrchestrator:
    """Multi-Agent Orchestration System"""
    
    def __init__(self):
        self.telemedicine = TelemedicineAI()
        self.agriculture = AgricultureAI()
        self.finance = FinanceAI()
        self.decisions_log = []
        
    def coordinate_farmer_health_emergency(self, farmer_data):
        """Coordinate response to farmer health emergency"""
        print("🚨 EMERGENCY SCENARIO: Farmer Health Crisis")
        print("=" * 50)
        
        # Step 1: Medical Assessment
        medical_result = self.telemedicine.diagnose_patient(farmer_data['health_data'])
        print(f"🏥 Medical AI: {medical_result['diagnosis']} (Confidence: {medical_result['confidence']:.2f})")
        
        # Step 2: Process Insurance Claim
        if medical_result['diagnosis'] != "Healthy":
            estimated_cost = np.random.uniform(200, 1500)
            insurance_result = self.telemedicine.process_insurance_claim(
                farmer_data['farmer_id'], 
                medical_result['diagnosis'], 
                estimated_cost
            )
            print(f"💰 Insurance: {'Approved' if insurance_result['claim_approved'] else 'Denied'} - ${estimated_cost:.2f}")
        
        # Step 3: Assess Farm Impact
        crop_result = self.agriculture.assess_crop_health(farmer_data['field_data'])
        print(f"🌾 Agriculture AI: Crop health {crop_result['health_score']:.1f}/100 - {crop_result['status']}")
        
        # Step 4: Financial Adjustment
        if medical_result['diagnosis'] == "Emergency":
            # Temporarily adjust credit terms
            credit_adjustment = self.finance.calculate_credit_score(farmer_data['financial_data'])
            print(f"📊 Finance AI: Credit score {credit_adjustment['credit_score']} - Emergency support activated")
        
        # Step 5: Coordinated Decision
        decision = self._make_coordinated_decision(medical_result, crop_result, farmer_data)
        self.decisions_log.append(decision)
        
        print(f"🤖 AI Coordinator: {decision['action']}")
        print()
        
        return decision
    
    def coordinate_weather_event(self, weather_data):
        """Coordinate response to severe weather event"""
        print("⛈️  WEATHER EVENT: Severe Storm Predicted")
        print("=" * 50)
        
        # Agriculture impact assessment
        for i, farm in enumerate(weather_data['affected_farms']):
            crop_assessment = self.agriculture.assess_crop_health(farm)
            print(f"🌾 Farm {i+1}: Health {crop_assessment['health_score']:.1f}/100")
            
            # Trigger insurance if needed
            if crop_assessment['health_score'] < 60:
                insurance = self.agriculture.process_crop_insurance(i+1, 'wheat', 50000)
                print(f"🛡️  Insurance activated: Premium ${insurance['premium']:.2f}")
        
        print("🤖 AI Coordinator: All farms prepared for weather event")
        print()
    
    def _make_coordinated_decision(self, medical, crop, farmer_data):
        """Make integrated decision across all domains"""
        priority_score = 0
        actions = []
        
        if medical['diagnosis'] == "Emergency":
            priority_score += 10
            actions.append("Immediate medical evacuation arranged")
        elif medical['diagnosis'] == "Needs Consultation":
            priority_score += 5
            actions.append("Telemedicine consultation scheduled")
        
        if crop['health_score'] < 70:
            priority_score += 3
            actions.append("Farm management assistant deployed")
        
        return {
            'priority_score': priority_score,
            'action': "; ".join(actions) if actions else "Continue monitoring",
            'timestamp': datetime.now()
        }

def run_comprehensive_demo():
    """Run the complete AgriMedFinance AI demonstration"""
    
    print("🌟 AGRIMEDFINANCE AI ECOSYSTEM DEMO")
    print("=" * 60)
    print()
    
    # Initialize the orchestrator
    orchestrator = AgenticOrchestrator()
    
    # Demo Scenario 1: Farmer Health Emergency
    farmer_emergency_data = {
        'farmer_id': 'F001',
        'health_data': {
            'age': 45,
            'symptoms_count': 7,
            'blood_pressure': 180,
            'lab_results': 85
        },
        'field_data': {
            'soil_moisture': 0.6,
            'temperature': 28,
            'rainfall': 45,
            'fertilizer_used': 75
        },
        'financial_data': {
            'income': 65000,
            'debt_ratio': 0.4,
            'payment_history': 0.9,
            'assets': 180000
        }
    }
    
    result1 = orchestrator.coordinate_farmer_health_emergency(farmer_emergency_data)
    
    # Demo Scenario 2: Weather Event
    weather_event_data = {
        'event_type': 'severe_storm',
        'affected_farms': [
            {'soil_moisture': 0.3, 'temperature': 32, 'rainfall': 5, 'fertilizer_used': 60},
            {'soil_moisture': 0.5, 'temperature': 30, 'rainfall': 15, 'fertilizer_used': 80},
            {'soil_moisture': 0.7, 'temperature': 25, 'rainfall': 25, 'fertilizer_used': 70}
        ]
    }
    
    orchestrator.coordinate_weather_event(weather_event_data)
    
    # Individual Module Demonstrations
    print("📊 INDIVIDUAL MODULE DEMONSTRATIONS")
    print("=" * 50)
    
    # Telemedicine Demo
    print("🏥 TELEMEDICINE AI:")
    patient_data = {'age': 35, 'symptoms_count': 3, 'blood_pressure': 140, 'lab_results': 75}
    diagnosis = orchestrator.telemedicine.diagnose_patient(patient_data)
    print(f"    Diagnosis: {diagnosis['diagnosis']} ({diagnosis['confidence']:.2f} confidence)")
    print(f"    Recommendation: {diagnosis['recommendation']}")
    print()
    
    # Agriculture Demo
    print("🌾 AGRICULTURE AI:")
    field_data = {'soil_moisture': 0.4, 'temperature': 30, 'rainfall': 20, 'fertilizer_used': 65}
    crop_health = orchestrator.agriculture.assess_crop_health(field_data)
    yield_pred = orchestrator.agriculture.predict_yield({})
    print(f"    Crop Health: {crop_health['health_score']:.1f}/100 - {crop_health['status']}")
    print(f"    Predicted Yield: {yield_pred['predicted_yield']} tons/hectare")
    print()
    
    # Finance Demo
    print("💰 FINANCE AI:")
    applicant_data = {'income': 55000, 'debt_ratio': 0.3, 'payment_history': 0.85, 'assets': 120000}
    credit_score = orchestrator.finance.calculate_credit_score(applicant_data)
    fraud_check = orchestrator.finance.detect_fraud({})
    print(f"    Credit Score: {credit_score['credit_score']} ({credit_score['rating']})")
    print(f"    Fraud Risk: {fraud_check['risk_level']} ({fraud_check['fraud_probability']:.3f})")
    print()
    
    # Smart Contract Summary
    print("⛓️  SMART CONTRACT TRANSACTIONS:")
    print("-" * 30)
    all_contracts = [
        orchestrator.telemedicine.contract,
        orchestrator.agriculture.contract,
        orchestrator.finance.contract
    ]
    
    total_transactions = 0
    total_gas = 0
    
    for contract in all_contracts:
        print(f"{contract.contract_type}: {len(contract.transactions)} transactions")
        for tx in contract.transactions[-2:]:  # Show last 2 transactions
            print(f"  ├─ {tx['function']}: {tx['tx_hash']} (Gas: {tx['gas_used']})")
        total_transactions += len(contract.transactions)
        total_gas += sum(tx['gas_used'] for tx in contract.transactions)
    
    print(f"\nTotal: {total_transactions} transactions, {total_gas:,} gas used")
    print()
    
    # Performance Visualization
    create_performance_dashboard(orchestrator)
    
    # Smart Contract Integration Demo
    demonstrate_smart_contract_integration()
    
    return orchestrator

def create_performance_dashboard(orchestrator):
    """Create visualization dashboard"""
    
    # Create sample performance data
    modules = ['Telemedicine', 'Agriculture', 'Finance']
    accuracy_scores = [0.89, 0.92, 0.86]
    response_times = [0.3, 0.5, 0.2]  # seconds
    gas_costs = [45000, 52000, 38000]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('AgriMedFinance AI System Performance Dashboard', fontsize=16, fontweight='bold')
    
    # AI Model Accuracy
    bars1 = ax1.bar(modules, accuracy_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('AI Model Accuracy Scores')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0.8, 1.0)
    for i, v in enumerate(accuracy_scores):
        ax1.text(i, v + 0.005, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Response Times
    bars2 = ax2.bar(modules, response_times, color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax2.set_title('Average Response Times')
    ax2.set_ylabel('Time (seconds)')
    for i, v in enumerate(response_times):
        ax2.text(i, v + 0.01, f'{v}s', ha='center', fontweight='bold')
    
    # Smart Contract Gas Usage
    bars3 = ax3.bar(modules, gas_costs, color=['#FFB347', '#87CEEB', '#98FB98'])
    ax3.set_title('Smart Contract Gas Usage')
    ax3.set_ylabel('Gas (units)')
    for i, v in enumerate(gas_costs):
        ax3.text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')
    
    # Decision Timeline
    decisions_per_hour = [12, 15, 8, 18, 22, 16, 14]
    hours = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00']
    ax4.plot(hours, decisions_per_hour, marker='o', linewidth=3, markersize=8, color='#FF6B6B')
    ax4.fill_between(hours, decisions_per_hour, alpha=0.3, color='#FF6B6B')
    ax4.set_title('AI Decisions Over Time')
    ax4.set_ylabel('Decisions per Hour')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Create integration flow diagram
    create_integration_flow()

def create_integration_flow():
    """Visualize the AI agent integration flow"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Sample data for integration metrics
    integration_data = {
        'Cross-Domain Decisions': 25,
        'Automated Contracts': 42,
        'Conflict Resolutions': 8,
        'Emergency Responses': 3,
        'Preventive Actions': 17
    }
    
    # Create a more detailed visualization
    categories = list(integration_data.keys())
    values = list(integration_data.values())
    
    # Create horizontal bar chart
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    bars = ax.barh(categories, values, color=colors)
    
    # --- THIS IS THE CORRECTED LINE ---
    ax.set_title('AI Agent Integration Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Operations', fontsize=12)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                str(value), ha='left', va='center', fontweight='bold')
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()

def demonstrate_smart_contract_integration():
    """Demonstrate smart contract integration with AI decisions"""
    
    print("⛓️  SMART CONTRACT + AI INTEGRATION DEMO")
    print("=" * 60)
    
    # Create sample contract interactions
    contracts_data = []
    
    # Medical insurance contracts
    for i in range(5):
        contracts_data.append({
            'contract_type': 'Medical Insurance',
            'ai_decision': f'Auto-approve claim ${np.random.randint(100, 800)}',
            'gas_used': np.random.randint(35000, 65000),
            'execution_time': np.random.uniform(0.1, 0.8),
            'success': True
        })
    
    # Agricultural insurance contracts
    for i in range(4):
        contracts_data.append({
            'contract_type': 'Crop Insurance',
            'ai_decision': f'Weather damage claim ${np.random.randint(1000, 5000)}',
            'gas_used': np.random.randint(40000, 70000),
            'execution_time': np.random.uniform(0.2, 1.0),
            'success': True
        })
    
    # Lending contracts
    for i in range(6):
        contracts_data.append({
            'contract_type': 'Smart Lending',
            'ai_decision': f'Loan approval ${np.random.randint(5000, 25000)}',
            'gas_used': np.random.randint(30000, 55000),
            'execution_time': np.random.uniform(0.15, 0.6),
            'success': np.random.choice([True, True, True, False])  # 75% success rate
        })
    
    # Create DataFrame for analysis
    df = pd.DataFrame(contracts_data)
    
    print("📊 Contract Execution Summary:")
    print(f"    Total Contracts: {len(df)}")
    print(f"    Success Rate: {df['success'].mean():.1%}")
    print(f"    Average Gas Used: {df['gas_used'].mean():,.0f}")
    print(f"    Average Execution Time: {df['execution_time'].mean():.2f}s")


# --- Main Execution ---
if __name__ == '__main__':
    # To run this in a script or notebook, simply call the main function.
    # The output will be printed to the console and plots will be displayed.
    orchestrator_instance = run_comprehensive_demo()