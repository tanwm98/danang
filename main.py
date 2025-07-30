import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import gradio as gr
from PIL import Image

# 1. CUSTOM CNN TRAINING SCRIPT
class CustomPestClassifier:
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
    
    def build_custom_cnn(self):
        """Build custom CNN from scratch - no pre-trained weights"""
        
        print("🏗️ Building custom CNN from scratch...")
        
        self.model = keras.Sequential([
            # Input layer
            layers.Input(shape=(*self.img_size, 3)),
            
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Conv Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Custom CNN built with {self.model.count_params():,} parameters")
        
        # Print model architecture
        print("\n📋 Model Architecture:")
        self.model.summary()
        
        return self.model
    
    def train_from_scratch(self, epochs=20):
        """Train custom CNN from scratch (needs more epochs)"""
        
        print(f"🚀 Training custom CNN from scratch for {epochs} epochs...")
        print("⏰ This will take longer since we're not using pre-trained weights")
        
        # More comprehensive callbacks for scratch training
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=5, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'custom_pest_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            )
        ]
        
        # Train
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model.save('custom_pest_model.h5')
        print("✅ Custom model saved as 'custom_pest_model.h5'")
        
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
class CustomPestApp:
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
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        
        # Results
        result = f"**🎯 Primary Detection:** {predicted_pest} ({confidence:.1%} confidence)\n\n"
        result += "**📊 Top 3 Predictions:**\n"
        for i, idx in enumerate(top_3_idx, 1):
            pest = self.class_names[idx]
            conf = predictions[0][idx]
            result += f"{i}. {pest}: {conf:.1%}\n"
        
        treatment = f"**🌿 Organic Treatment:** {TREATMENTS.get(predicted_pest, 'No treatment info')}"
        
        return result, treatment
    
    def create_interface(self):
        """Simple Gradio interface"""
        
        with gr.Blocks(title="🐛 Custom CNN Pest Identifier") as app:
            gr.Markdown("""
            # 🐛 Custom CNN Pest Identifier
            ## 🧠 Built from Scratch - No Pre-trained Weights!
            
            Upload a pest photo to get identification and organic treatment advice.
            """)
            
            with gr.Row():
                image_input = gr.Image(label="📸 Upload Pest Photo", type="pil")
                
                with gr.Column():
                    result_output = gr.Markdown(label="🔍 Detection Results")
                    treatment_output = gr.Markdown(label="🌿 Treatment Advice")
            
            identify_btn = gr.Button("🔍 Identify Pest", variant="primary", size="lg")
            
            identify_btn.click(
                fn=self.predict,
                inputs=[image_input],
                outputs=[result_output, treatment_output]
            )
            
            gr.Markdown("""
            ---
            ### 📋 Supported Pest Types
            Ants • Bees • Beetles • Caterpillars • Earthworms • Earwigs • 
            Grasshoppers • Moths • Slugs • Snails • Wasps • Weevils
            
            ### 🧠 Model Info
            - **Architecture**: Custom CNN built from scratch
            - **No Transfer Learning**: Trained entirely on your pest dataset
            - **Input Size**: 160x160 pixels
            """)
        
        return app

# 4. SIMPLE USAGE FUNCTIONS

def train_custom():
    """Train custom CNN from scratch"""
    
    print("🚀 Custom CNN Training from Scratch")
    print("📝 No pre-trained weights - building everything from ground up!")
    
    # Initialize
    classifier = CustomPestClassifier(data_dir="dataset")
    
    # Prepare data
    print("📊 Preparing data...")
    train_ds, val_ds = classifier.prepare_data()
    
    # Build custom model
    print("🏗️ Building custom CNN...")
    model = classifier.build_custom_cnn()
    
    # Train from scratch (needs more epochs)
    history = classifier.train_from_scratch(epochs=20)
    
    print("✅ Custom training complete!")
    return history

def launch_custom_app():
    """Launch custom CNN app"""
    
    import os
    if not os.path.exists('custom_pest_model.h5'):
        print("❌ No custom model found! Run train_custom() first.")
        return
    
    print("🚀 Launching Custom CNN Pest App...")
    print("   🧠 Using your custom-built CNN (no pre-trained weights)")
    print("   🌐 App will open in your default browser")
    print("   📱 Upload pest images to test!")
    
    app = CustomPestApp('custom_pest_model.h5')
    interface = app.create_interface()
    interface.launch(share=True)

def check_custom_model_status():
    """Check if custom trained model exists and show info"""
    
    import os
    from datetime import datetime
    
    if os.path.exists('custom_pest_model.h5'):
        model_size = os.path.getsize('custom_pest_model.h5') / (1024*1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime('custom_pest_model.h5'))
        
        print("✅ Custom CNN model found!")
        print(f"   🧠 Type: Custom CNN (built from scratch)")
        print(f"   📊 Size: {model_size:.1f}MB")
        print(f"   📅 Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("   🚀 Ready to launch app!")
        return True
    else:
        print("❌ No custom model found")
        print("   🏗️ Need to run custom training first")
        return False

def run_custom_pipeline(force_retrain=False):
    """Smart pipeline for custom CNN: Check for model, train if needed, then launch"""
    
    import os
    from datetime import datetime
    
    print("🎯 Custom CNN Pipeline: Check Model → Train (if needed) → Launch App")
    
    # Check if model already exists
    if os.path.exists('custom_pest_model.h5') and not force_retrain:
        # Get model info
        model_size = os.path.getsize('custom_pest_model.h5') / (1024*1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime('custom_pest_model.h5'))
        
        print(f"✅ Found existing custom CNN model 'custom_pest_model.h5'")
        print(f"   🧠 Type: Custom CNN (no pre-trained weights)")
        print(f"   📊 Size: {model_size:.1f}MB")
        print(f"   📅 Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("⏭️ Skipping training - launching app directly...")
        print("   💡 Tip: Use force_retrain=True to retrain anyway")
        
        launch_custom_app()
    else:
        if force_retrain and os.path.exists('custom_pest_model.h5'):
            print("🔄 Force retrain requested - training new custom model...")
        else:
            print("❌ No existing custom model found")
        
        print("🚀 Training new custom CNN from scratch...")
        print("⏰ This will take longer (~20 epochs) since no pre-trained weights")
        train_custom()
        print("🚀 Training complete! Launching app...")
        launch_custom_app()

def visualize_model_architecture():
    """Show the custom CNN architecture"""
    
    print("🧠 Creating model to show architecture...")
    classifier = CustomPestClassifier(data_dir="dataset")
    model = classifier.build_custom_cnn()
    
    print("\n" + "="*60)
    print("🏗️ CUSTOM CNN ARCHITECTURE")
    print("="*60)
    print("Built from scratch - no pre-trained weights!")
    print("="*60)
    
    # Show detailed summary
    model.summary()
    
    # Show layer breakdown
    print("\n📊 Layer Breakdown:")
    conv_layers = 0
    dense_layers = 0
    dropout_layers = 0
    
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            conv_layers += 1
        elif isinstance(layer, layers.Dense):
            dense_layers += 1
        elif isinstance(layer, layers.Dropout):
            dropout_layers += 1
    
    print(f"   • Convolutional layers: {conv_layers}")
    print(f"   • Dense layers: {dense_layers}")
    print(f"   • Dropout layers: {dropout_layers}")
    print(f"   • Total parameters: {model.count_params():,}")
    print(f"   • Trainable parameters: {model.count_params():,} (all of them!)")

# CUSTOM CNN USAGE
if __name__ == "__main__":
    # Smart pipeline for custom CNN - automatically detects existing model!
    run_custom_pipeline()
    
    # Other options:
    # visualize_model_architecture()          # Show model architecture
    # check_custom_model_status()             # Check if custom model exists
    # run_custom_pipeline(force_retrain=True) # Force retrain custom CNN
    # launch_custom_app()                     # Just launch app
    # train_custom()                          # Just train custom model