import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import gradio as gr
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import tempfile
import shutil
from sklearn.metrics import classification_report
from datetime import datetime
import requests
import json


# 1. MOBILENETV2 TRAINING SCRIPT (unchanged)
class CustomPestClassifier:
    def __init__(self, data_dir, img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.class_names = sorted(
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        )
        self.model = None
    
    def prepare_data(self, batch_size=32, train_split=0.6, val_split=0.2, test_split=0.2):
        """Enhanced data preparation with train-validation-test splits"""
        
        # Verify splits add up to 1.0
        assert abs(train_split + val_split + test_split - 1.0) < 0.001, "Splits must add up to 1.0"
        
        print(f"📊 Data splits: Train {train_split*100:.0f}% | Validation {val_split*100:.0f}% | Test {test_split*100:.0f}%")
        
        # Less aggressive data augmentation for transfer learning
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.15,
        )
        # No augmentation for validation and test
        val_test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        # Collect all file paths and labels
        all_paths = []
        all_labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_paths.append(os.path.join(class_dir, img_file))
                        all_labels.append(class_idx)
        
        print(f"📁 Found {len(all_paths)} total images across {len(self.class_names)} classes")
        
        # First split: separate test set
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            all_paths, all_labels, 
            test_size=test_split, 
            stratify=all_labels,  # Ensure balanced split
            random_state=42
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_split / (train_split + val_split)  # Adjust for remaining data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=val_size_adjusted,
            stratify=train_val_labels,
            random_state=42
        )
        
        print(f"✅ Final splits: Train={len(train_paths)} | Val={len(val_paths)} | Test={len(test_paths)}")
        
        self.temp_dir = tempfile.mkdtemp()
        train_dir = os.path.join(self.temp_dir, 'train')
        val_dir = os.path.join(self.temp_dir, 'val') 
        test_dir = os.path.join(self.temp_dir, 'test')
        
        # Create class subdirectories
        for split_dir in [train_dir, val_dir, test_dir]:
            for class_name in self.class_names:
                os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
        
        # Copy files to appropriate directories
        def copy_files(paths, labels, target_dir):
            for path, label in zip(paths, labels):
                class_name = self.class_names[label]
                target_path = os.path.join(target_dir, class_name, os.path.basename(path))
                shutil.copy2(path, target_path)
        
        print("📋 Creating temporary split directories...")
        copy_files(train_paths, train_labels, train_dir)
        copy_files(val_paths, val_labels, val_dir)
        copy_files(test_paths, test_labels, test_dir)
        
        # Create data generators
        self.train_ds = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        self.val_ds = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        self.test_ds = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        return self.train_ds, self.val_ds, self.test_ds
    
    def build_mobilenetv2_model(self):
        """Build MobileNetV2 transfer learning model"""
        
        print("🏗️ Building MobileNetV2 transfer learning model...")
        
        # Load pre-trained MobileNetV2
        base_model = keras.applications.MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'  # Pre-trained weights
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom head
        self.model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.002),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ MobileNetV2 model built with {self.model.count_params():,} parameters")
        print(f"   🔒 Frozen parameters: {base_model.count_params():,}")
        print(f"   🔓 Trainable parameters: {self.model.count_params() - base_model.count_params():,}")
        
        # Print model architecture
        print("\n📋 Model Architecture:")
        self.model.summary()
        
        return self.model
    
    def train_from_scratch(self, epochs=20):
        """Train MobileNetV2 with transfer learning"""
        
        print(f"🚀 Training MobileNetV2 for {epochs} epochs...")
        print("⏰ Transfer learning with MobileNetV2 + ImageNet weights")
        
        # Better callbacks for transfer learning
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True,
                monitor='val_accuracy',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'mobilenetv2_pest_model.keras',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            )
        ]
        
        # Train using train and validation sets
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model.save('mobilenetv2_pest_model.keras')
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(self.test_ds, verbose=1)
        
        # Get detailed test predictions
        test_predictions = self.model.predict(self.test_ds, verbose=1)
        
        # Print comprehensive results
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        print(f"\n📈 FINAL RESULTS:")
        print(f"   Training Accuracy:   {final_train_acc:.3f} ({final_train_acc*100:.1f}%)")
        print(f"   Validation Accuracy: {final_val_acc:.3f} ({final_val_acc*100:.1f}%)")
        print(f"   Test Accuracy:       {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
        print(f"\n📊 PERFORMANCE ANALYSIS:")
        print(f"   Train-Val Gap:       {final_train_acc - final_val_acc:.3f} ({'Overfitting' if final_train_acc - final_val_acc > 0.1 else 'Good'})")
        print(f"   Val-Test Gap:        {final_val_acc - test_accuracy:.3f} ({'Consistent' if abs(final_val_acc - test_accuracy) < 0.05 else 'Inconsistent'})")
        print(f"   Generalization:      {'Good' if test_accuracy > 0.7 else 'Fair' if test_accuracy > 0.5 else 'Poor'}")
        
        # Per-class test accuracy (if possible)
        try:
            # Get true labels
            test_labels = []
            for i in range(len(self.test_ds)):
                batch = self.test_ds[i]
                test_labels.extend(np.argmax(batch[1], axis=1))
                if len(test_labels) >= self.test_ds.samples:
                    test_labels = test_labels[:self.test_ds.samples]
                    break
            
            # Get predicted labels
            pred_labels = np.argmax(test_predictions[:len(test_labels)], axis=1)
            
            print(f"\n📋 PER-CLASS TEST PERFORMANCE:")
            report = classification_report(test_labels, pred_labels, 
                                         target_names=self.class_names, 
                                         output_dict=True)
            
            for class_name in self.class_names:
                if class_name in report:
                    acc = report[class_name]['precision']
                    print(f"   {class_name:12}: {acc:.3f} ({acc*100:.1f}%)")
                    
        except Exception as e:
            print(f"   (Detailed per-class analysis failed: {e})")
        
        # Cleanup temporary directories
        self.cleanup_temp_dirs()
        
        print(f"\n🎯 CONCLUSION:")
        if test_accuracy >= 0.75:
            print("   🎉 Excellent performance! Model generalizes well.")
        elif test_accuracy >= 0.6:
            print("   ✅ Good performance! Model is working well.")
        elif test_accuracy >= 0.45:
            print("   ⚠️ Fair performance. Consider more training or data.")
        else:
            print("   ❌ Poor performance. Model needs significant improvement.")
            
        return history, test_accuracy
    
    def cleanup_temp_dirs(self):
        """Clean up temporary directories"""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print("🧹 Cleaned up temporary directories")
        except Exception as e:
            print(f"⚠️ Could not clean up temp dirs: {e}")


# 2. OPENWEBUI API CLIENT WITH AUTHENTICATION
class OpenWebUIClient:
    def __init__(self, base_url="http://localhost:3000", model="pest-management", email=None, password=None):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.email = email
        self.password = password
        self.api_endpoint = f"{self.base_url}/ollama/api/chat"
        self.auth_endpoint = f"{self.base_url}/api/v1/auths/signin"
        self.token = None
        self.headers = {"Content-Type": "application/json"}
        
        # Authenticate if credentials provided
        if email and password:
            self.authenticate()
    
    def authenticate(self):
        """Authenticate with OpenWebUI and get JWT token"""
        auth_payload = {
            "email": self.email,
            "password": self.password
        }
        
        try:
            print(f"🔐 Authenticating with OpenWebUI as {self.email}...")
            
            response = requests.post(
                self.auth_endpoint,
                json=auth_payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                auth_data = response.json()
                self.token = auth_data.get('token')
                if self.token:
                    self.headers["Authorization"] = f"Bearer {self.token}"
                    print("✅ Authentication successful!")
                    return True
                else:
                    print("❌ No token received in response")
                    return False
            else:
                print(f"❌ Authentication failed (HTTP {response.status_code}): {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Authentication error: {str(e)}")
            return False
    
    def get_pest_treatment(self, pest_name, confidence, top_predictions):
        """Query OpenWebUI for pest treatment advice"""
        
        # Construct prompt for the pest management model
        prompt = f"""I've identified a pest in my garden with the following details:

Primary Detection: {pest_name}
Confidence Level: {confidence:.1%}

Top 3 possibilities:
{chr(10).join([f"{i+1}. {pred['name']}: {pred['confidence']:.1%}" for i, pred in enumerate(top_predictions)])}

Please provide comprehensive organic garden pest management advice including:
- What this pest is and its garden impact
- Whether it's beneficial or harmful
- Safe, organic treatment options
- Best timing for treatment
- Safety considerations for humans and beneficial insects
- Prevention strategies

Focus on environmentally friendly, organic methods only. If this is a beneficial insect, emphasize protection rather than elimination."""

        # Prepare API request
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }
        
        try:
            print(f"🌐 Querying OpenWebUI at {self.api_endpoint}")
            print(f"   🤖 Model: {self.model}")
            print(f"   🐛 Pest: {pest_name} ({confidence:.1%} confidence)")
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=30,
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'message' in result and 'content' in result['message']:
                    return result['message']['content']
                else:
                    return f"❌ Unexpected response format from OpenWebUI: {result}"
            else:
                return f"❌ OpenWebUI API error (HTTP {response.status_code}): {response.text}"
                
        except requests.exceptions.ConnectionError:
            return f"""❌ **Could not connect to OpenWebUI**

Please ensure:
- OpenWebUI is running at {self.base_url}
- The pest-management model is available
- No firewall blocking the connection

**Fallback:** Try accessing {self.base_url} directly in your browser to verify it's running."""

        except requests.exceptions.Timeout:
            return "❌ **Request timed out**\n\nThe OpenWebUI API took too long to respond. This might be because the model is loading or the system is busy."

        except Exception as e:
            return f"❌ **Unexpected error:** {str(e)}"


# 3. UPDATED PEST APP WITH OPENWEBUI INTEGRATION
class CustomPestApp:
    def __init__(self, model_path, openwebui_url="http://localhost:3000", openwebui_model="pest-management", 
                 email=None, password=None):
        self.model = keras.models.load_model(model_path)
        self.class_names = ['Ants', 'Bees', 'Beetles', 'Caterpillars', 'Earthworms', 
                           'Earwigs', 'Grasshoppers', 'Moths', 'Slugs', 'Snails', 
                           'Wasps', 'Weevils']
        self.openwebui_client = OpenWebUIClient(openwebui_url, openwebui_model, email, password)
    
    def predict(self, image):
        """Enhanced prediction with OpenWebUI integration"""
        if image is None:
            return "Please upload an image", "Upload an image to get treatment recommendations"
        
        # Preprocess - MUST match training size!
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_pest = self.class_names[predicted_idx]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = []
        
        for idx in top_3_idx:
            top_predictions.append({
                'name': self.class_names[idx],
                'confidence': predictions[0][idx]
            })
        
        # Build identification results
        result = f"# 🎯 Pest Identification Results\n\n"
        result += f"**Primary Detection:** {predicted_pest}\n"
        
        # Confidence level indicator
        if confidence >= 0.8:
            conf_indicator = "🟢 High Confidence"
        elif confidence >= 0.6:
            conf_indicator = "🟡 Medium Confidence"
        else:
            conf_indicator = "🔴 Low Confidence"
        
        result += f"**Confidence:** {confidence:.1%} {conf_indicator}\n\n"
        
        result += "### 📊 Top 3 Possibilities:\n"
        for i, pred in enumerate(top_predictions, 1):
            result += f"{i}. **{pred['name']}**: {pred['confidence']:.1%}\n"
        
        # Query OpenWebUI for treatment advice
        treatment_info = self.openwebui_client.get_pest_treatment(
            predicted_pest, confidence, top_predictions
        )
        
        return result, treatment_info
    
    def create_interface(self):
        """Enhanced Gradio interface with OpenWebUI integration"""
        
        with gr.Blocks(
            title="🐛 Smart Garden Pest Identifier with AI Assistant",
            theme=gr.themes.Default()
        ) as app:
            
            gr.Markdown(f"""
            # 🐛 Smart Garden Pest Identifier
            ## 🧠 AI-Powered Pest Recognition & Dynamic Treatment Advisor
            
            **Simply upload a photo of any garden pest to instantly get:**
            - 🎯 Accurate pest identification using MobileNetV2
            - 🤖 AI-powered treatment recommendations from OpenWebUI
            - 🌿 Personalized organic treatment strategies
            - ⏰ Optimal timing and safety guidance
            
            **🔗 Connected to:** `{self.openwebui_client.base_url}` | **🤖 Model:** `{self.openwebui_client.model}`
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="📸 Upload Pest Photo", 
                        type="pil"
                    )
                    
                    identify_btn = gr.Button(
                        "🔍 Identify Pest & Get AI Treatment Advice", 
                        variant="primary"
                    )
                    
                    gr.Markdown("""
                    **📋 Tips for Best Results:**
                    - 📷 Take clear, close-up photos
                    - 🔆 Use good lighting
                    - 🎯 Center the pest in the image
                    - 📏 Include size reference if possible
                    
                    **🤖 AI Treatment Advisor:**
                    - Dynamic responses from OpenWebUI
                    - Personalized to your specific situation
                    - Always organic and eco-friendly methods
                    """)
                
                with gr.Column(scale=2):
                    result_output = gr.Markdown(
                        value="Upload an image to see identification results here..."
                    )
                    
                    treatment_output = gr.Markdown(
                        value="AI treatment recommendations will appear here after identification..."
                    )
            
            # Connect the button to the function
            identify_btn.click(
                fn=self.predict,
                inputs=[image_input],
                outputs=[result_output, treatment_output]
            )
            
            gr.Markdown("""
            ---
            ## 🌱 Supported Garden Pest Types
            
            **🐛 Common Pests:** Ants • Beetles • Caterpillars • Earwigs • Grasshoppers • Moths • Slugs • Snails • Weevils
            
            **🌟 Beneficial Insects:** Bees • Earthworms • Wasps *(AI will tell you how to protect these garden helpers!)*
            
            ## 🧠 Technology Stack
            - **Vision AI:** MobileNetV2 with ImageNet Transfer Learning
            - **Treatment AI:** OpenWebUI with specialized pest-management model
            - **Training:** 1.4M ImageNet images + Custom pest dataset
            - **Focus:** Organic, safe, environmentally responsible methods
            
            ---
            *🌿 Powered by AI • Always organic • Beneficial insect friendly*
            """)
        
        return app


# 4. USAGE FUNCTIONS (mostly unchanged)
def train_mobilenetv2():
    """Train MobileNetV2 transfer learning model"""
    
    print("🚀 MobileNetV2 Transfer Learning Training")
    print("📝 Using ImageNet pre-trained weights for better accuracy!")
    
    # Initialize
    classifier = CustomPestClassifier(data_dir="dataset")
    
    # Prepare data with train-val-test splits
    print("📊 Preparing data with train-validation-test splits...")
    train_ds, val_ds, test_ds = classifier.prepare_data()
    
    # Build MobileNetV2 model
    print("🏗️ Building MobileNetV2 model...")
    model = classifier.build_mobilenetv2_model()
    
    # Train with proper evaluation (includes test set evaluation)
    history, test_accuracy = classifier.train_from_scratch(epochs=20)
    
    print("✅ MobileNetV2 training with test evaluation complete!")
    print(f"🎯 Final test accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    
    return history, test_accuracy

def launch_mobilenetv2_app(openwebui_url="http://localhost:3000", openwebui_model="pest-management",
                          email=None, password=None):
    """Launch MobileNetV2 app with OpenWebUI integration"""
    
    if not os.path.exists('mobilenetv2_pest_model.keras'):
        print("❌ No MobileNetV2 model found! Run train_mobilenetv2() first.")
        return
    
    print("🚀 Launching MobileNetV2 Pest App with OpenWebUI Integration...")
    print("   🧠 Vision: MobileNetV2 + ImageNet transfer learning")
    print(f"   🤖 AI Assistant: {openwebui_url} (model: {openwebui_model})")
    if email:
        print(f"   🔐 User: {email}")
    print("   🌐 App will open in your default browser")
    print("   📱 Upload pest images to test!")
    
    app = CustomPestApp('mobilenetv2_pest_model.keras', openwebui_url, openwebui_model, email, password)
    interface = app.create_interface()
    interface.launch(share=True)

def run_mobilenetv2_pipeline(force_retrain=False, openwebui_url="http://localhost:3000", 
                            openwebui_model="pest-management", email=None, password=None):
    """Smart pipeline for MobileNetV2 with OpenWebUI integration"""
    
    print("🎯 MobileNetV2 + OpenWebUI Pipeline: Check Model → Train-Val-Test → Launch App")
    
    # Check if model already exists
    if os.path.exists('mobilenetv2_pest_model.keras') and not force_retrain:
        # Get model info
        model_size = os.path.getsize('mobilenetv2_pest_model.keras') / (1024*1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime('mobilenetv2_pest_model.keras'))
        
        print(f"✅ Found existing MobileNetV2 model 'mobilenetv2_pest_model.keras'")
        print(f"   🧠 Type: MobileNetV2 Transfer Learning")
        print(f"   📊 Size: {model_size:.1f}MB")
        print(f"   📅 Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("⏭️ Skipping training - launching app with OpenWebUI integration...")
        print("   💡 Tip: Use force_retrain=True to retrain anyway")
        
        launch_mobilenetv2_app(openwebui_url, openwebui_model, email, password)
    else:
        if force_retrain and os.path.exists('mobilenetv2_pest_model.keras'):
            print("🔄 Force retrain requested - training new MobileNetV2 model...")
        else:
            print("❌ No existing MobileNetV2 model found")
        
        try:
            history, test_accuracy = train_mobilenetv2()
            print(f"\n🎉 Training complete! Final test accuracy: {test_accuracy*100:.1f}%")
            print("🚀 Launching app with OpenWebUI integration...")
            launch_mobilenetv2_app(openwebui_url, openwebui_model, email, password)
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("💡 Make sure you have required packages: pip install scikit-learn requests")

# MAIN USAGE
if __name__ == "__main__":
    # Run with authentication (replace with your credentials)
    run_mobilenetv2_pipeline(
        force_retrain=False,
        openwebui_url="http://localhost:3000",
        openwebui_model="pest-management",
        email="tanwm98@gmail.com",  # Your email
        password="admin"        # Your password
    )
    
    # Or run without credentials (will fail if auth required):
    # run_mobilenetv2_pipeline(force_retrain=False)
    
    # Just launch app with authentication:
    # launch_mobilenetv2_app(
    #     openwebui_url="http://localhost:3000",
    #     openwebui_model="pest-management",
    #     email="xxx@gmail.com",
    #     password="admin"
    # )