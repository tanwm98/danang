import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import gradio as gr
from PIL import Image

# 1. SIMPLE TRAINING SCRIPT
class SimplePestClassifier:
    def __init__(self, data_dir, img_size=(160, 160)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.class_names = ['Ants', 'Bees', 'Beetles', 'Caterpillars', 'Earthworms', 
                           'Earwigs', 'Grasshoppers', 'Moths', 'Slugs', 'Snails', 
                           'Wasps', 'Weevils']
        self.model = None
    
    def prepare_data(self, batch_size=32):
        """Simple data preparation"""
        
        # Basic data augmentation
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        self.train_ds = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        self.val_ds = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return self.train_ds, self.val_ds
    
    def build_mobilenetv2_model(self):
        """Build simple MobileNetV2 model"""
        
        # Load MobileNetV2
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model (no fine-tuning)
        base_model.trainable = False
        
        # Simple head
        self.model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Compile
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model built with {self.model.count_params():,} parameters")
        return self.model
    
    def train_simple(self, epochs=5):
        """Simple training - no fine-tuning, max 5 epochs"""
        
        print(f"🚀 Starting simple training for {epochs} epochs...")
        
        # Simple callback
        callbacks = [
            keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
        ]
        
        # Train
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        self.model.save('simple_pest_model.h5')
        print("✅ Model saved as 'simple_pest_model.h5'")
        
        return history

# 2. SIMPLE TREATMENT DATABASE
TREATMENTS = {
    'Ants': 'Use coffee grounds or cinnamon around plants',
    'Bees': 'PROTECT! Essential pollinators - do not treat',
    'Beetles': 'Apply neem oil spray or use row covers',
    'Caterpillars': 'Use Bt spray or hand pick',
    'Earthworms': 'BENEFICIAL! Improve soil - protect them',
    'Earwigs': 'Use newspaper traps or diatomaceous earth',
    'Grasshoppers': 'Use row covers or encourage birds',
    'Moths': 'Use pheromone traps or light traps',
    'Slugs': 'Use iron phosphate bait or copper strips',
    'Snails': 'Hand pick or use organic slug bait',
    'Wasps': 'BENEFICIAL predators - usually protect',
    'Weevils': 'Use beneficial nematodes or row covers'
}

# 3. SIMPLE GRADIO INTERFACE
class SimplePestApp:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.class_names = ['Ants', 'Bees', 'Beetles', 'Caterpillars', 'Earthworms', 
                           'Earwigs', 'Grasshoppers', 'Moths', 'Slugs', 'Snails', 
                           'Wasps', 'Weevils']
    
    def predict(self, image):
        """Simple prediction"""
        if image is None:
            return "Please upload an image", ""
        
        # Preprocess - MUST match training size!
        img = image.resize((160, 160))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_pest = self.class_names[predicted_idx]
        
        # Results
        result = f"**Detected:** {predicted_pest} ({confidence:.1%} confidence)"
        treatment = f"**Treatment:** {TREATMENTS.get(predicted_pest, 'No treatment info')}"
        
        return result, treatment
    
    def create_interface(self):
        """Simple Gradio interface"""
        
        with gr.Blocks(title="🐛 Simple Pest Identifier") as app:
            gr.Markdown("# 🐛 Simple Pest Identifier")
            
            with gr.Row():
                image_input = gr.Image(label="Upload Pest Photo", type="pil")
                
                with gr.Column():
                    result_output = gr.Markdown(label="Detection Result")
                    treatment_output = gr.Markdown(label="Treatment")
            
            identify_btn = gr.Button("Identify Pest", variant="primary")
            
            identify_btn.click(
                fn=self.predict,
                inputs=[image_input],
                outputs=[result_output, treatment_output]
            )
        
        return app

# 4. SIMPLE USAGE FUNCTIONS

def train_simple():
    """Simple training function"""
    
    print("🚀 Simple MobileNetV2 training (max 5 epochs, no fine-tuning)")
    
    # Initialize
    classifier = SimplePestClassifier(data_dir="dataset")
    
    # Prepare data
    print("📊 Preparing data...")
    train_ds, val_ds = classifier.prepare_data()
    
    # Build model
    print("🏗️ Building MobileNetV2...")
    model = classifier.build_mobilenetv2_model()
    
    # Train (max 5 epochs)
    history = classifier.train_simple(epochs=5)
    
    print("✅ Done!")
    return history

def check_model_status():
    """Check if trained model exists and show info"""
    
    import os
    from datetime import datetime
    
    if os.path.exists('simple_pest_model.h5'):
        model_size = os.path.getsize('simple_pest_model.h5') / (1024*1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime('simple_pest_model.h5'))
        
        print("✅ Trained model found!")
        print(f"   📊 Size: {model_size:.1f}MB")
        print(f"   📅 Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("   🚀 Ready to launch app!")
        return True
    else:
        print("❌ No trained model found")
        print("   🏃 Need to run training first")
        return False

def launch_simple_app():
    """Launch simple app"""
    
    import os
    if not os.path.exists('simple_pest_model.h5'):
        print("❌ No model found! Run train_simple() first.")
        return
    
    print("🚀 Launching Simple Pest App...")
    print("   🌐 App will open in your default browser")
    print("   📱 Upload pest images to test!")
    
    app = SimplePestApp('simple_pest_model.h5')
    interface = app.create_interface()
    interface.launch(share=True)

def run_simple_pipeline(force_retrain=False):
    """Smart pipeline: Check for model, train if needed, then launch"""
    
    import os
    from datetime import datetime
    
    print("🎯 Smart Pipeline: Check Model → Train (if needed) → Launch App")
    
    # Check if model already exists
    if os.path.exists('simple_pest_model.h5') and not force_retrain:
        # Get model info
        model_size = os.path.getsize('simple_pest_model.h5') / (1024*1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime('simple_pest_model.h5'))
        
        print(f"✅ Found existing model 'simple_pest_model.h5'")
        print(f"   📊 Size: {model_size:.1f}MB")
        print(f"   📅 Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("⏭️ Skipping training - launching app directly...")
        print("   💡 Tip: Use force_retrain=True to retrain anyway")
        
        launch_simple_app()
    else:
        if force_retrain and os.path.exists('simple_pest_model.h5'):
            print("🔄 Force retrain requested - training new model...")
        else:
            print("❌ No existing model found")
        
        print("🚀 Training new MobileNetV2 model...")
        train_simple()
        print("🚀 Training complete! Launching app...")
        launch_simple_app()

# SMART USAGE
if __name__ == "__main__":
    # Smart pipeline - automatically detects existing model!
    run_simple_pipeline()
    
    # Other options:
    # check_model_status()                    # Check if model exists
    # run_simple_pipeline(force_retrain=True) # Force retrain
    # launch_simple_app()                     # Just launch app
    # train_simple()                          # Just train model