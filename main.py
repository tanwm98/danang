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
import shutil


# 1. MOBILENETV2 TRAINING SCRIPT
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

# 2. SIMPLE TREATMENT DATABASE
# 2. ENHANCED TREATMENT DATABASE - More Human-Readable
TREATMENTS = {
    'Ants': {
        'description': '🐜 Small social insects commonly found in soil and on plants. They often protect aphids to harvest their sweet honeydew.',
        'impact': '⚠️ **Potential Problem**: May protect aphids and other harmful pests, indirectly damaging your crops.',
        'severity': 'Low to Medium',
        'beneficial': False,
        'treatments': {
            'immediate': [
                '☕ Sprinkle used coffee grounds around affected plants',
                '🌶️ Create a cinnamon barrier around plant bases',
                '🔥 Pour boiling water directly on ant trails (be careful not to damage plants)'
            ],
            'preventive': [
                '🌿 Apply food-grade diatomaceous earth around entry points',
                '🧽 Keep area clean of food debris',
                '💧 Fix any moisture problems that attract ants'
            ]
        },
        'when_to_treat': 'Only treat if ants are actively protecting aphids or other pests',
        'safety_notes': '✅ These methods are safe for humans, pets, and beneficial insects',
        'success_tips': 'Focus on breaking the trail rather than killing individual ants'
    },
    
    'Aphids': {
        'description': '🦟 Tiny, soft-bodied insects (green, black, or white) that cluster on leaf undersides and new growth.',
        'impact': '🚨 **High Threat**: Suck plant juices causing stunted growth, yellowing leaves, and can spread plant diseases.',
        'severity': 'High',
        'beneficial': False,
        'treatments': {
            'immediate': [
                '🚿 Blast off with strong water spray (repeat every 2-3 days)',
                '🧼 Apply insecticidal soap spray (2 tbsp dish soap per quart water)',
                '🌿 Spray neem oil solution in evening (avoid hot sunny days)'
            ],
            'biological': [
                '🐞 Release ladybugs (purchase from garden center)',
                '🦋 Attract lacewings with flowering plants nearby',
                '🌸 Plant companion flowers like marigolds and nasturtiums'
            ],
            'homemade': [
                '🧄 Garlic spray: blend 3 cloves garlic + 1 cup water, strain and spray',
                '🌶️ Chili spray: steep 1 tsp cayenne in 1 cup warm water for 30 minutes'
            ]
        },
        'when_to_treat': 'At first sign of infestation - check plants daily during spring/summer',
        'safety_notes': '⚠️ Apply treatments in evening to avoid harming beneficial insects',
        'success_tips': 'Consistency is key - treat every 3-5 days until population decreases'
    },
    
    'Bees': {
        'description': '🐝 Essential pollinators with fuzzy bodies and pollen baskets on their legs.',
        'impact': '🌟 **HERO INSECTS**: Absolutely crucial for pollinating crops and maintaining biodiversity.',
        'severity': 'BENEFICIAL',
        'beneficial': True,
        'treatments': {
            'support': [
                '🏡 Provide shallow water dishes with landing spots (stones/sticks)',
                '🌺 Plant bee-friendly flowers: sunflowers, lavender, wildflowers',
                '🚫 NEVER use pesticides when bees are active',
                '🍯 Contact local beekeepers if you find a swarm (they\'ll relocate safely)'
            ]
        },
        'when_to_treat': 'NEVER TREAT - Always protect and support!',
        'safety_notes': '🛡️ If allergic to bee stings, maintain distance but still protect them',
        'success_tips': 'A garden with happy bees is a productive garden!'
    },
    
    'Beetles': {
        'description': '🪲 Hard-shelled insects including cucumber beetles, Japanese beetles, and Colorado potato beetles.',
        'impact': '🚨 **Moderate to High Threat**: Chew holes in leaves, stems, fruits and can spread bacterial diseases.',
        'severity': 'Medium to High',
        'beneficial': False,
        'treatments': {
            'immediate': [
                '👋 Hand-pick beetles in early morning when they\'re sluggish',
                '🌿 Spray neem oil weekly (organic and safe)',
                '🥒 Use yellow sticky traps to catch flying beetles'
            ],
            'physical': [
                '🛡️ Cover young plants with floating row covers until flowering',
                '🌾 Use fine mesh or cheesecloth barriers',
                '💧 Shake beetles into soapy water bucket in morning'
            ],
            'cultural': [
                '🔄 Rotate crops yearly to break beetle life cycles',
                '🧹 Remove plant debris where beetles overwinter',
                '🌱 Plant trap crops (beetles prefer some plants over others)'
            ]
        },
        'when_to_treat': 'Start treatment at first beetle sighting - they multiply quickly',
        'safety_notes': '✅ Hand-picking and neem oil are completely safe methods',
        'success_tips': 'Early morning treatment is most effective when beetles are less active'
    },
    
    'Caterpillars': {
        'description': '🐛 Soft-bodied larvae of moths and butterflies, often green or striped, with chewing mouthparts.',
        'impact': '⚠️ **Variable Threat**: Can chew large holes in leaves and fruits, but some become beneficial butterflies.',
        'severity': 'Medium',
        'beneficial': 'Mixed',
        'treatments': {
            'targeted': [
                '🦠 Apply Bt spray (Bacillus thuringiensis) - targets only caterpillars, safe for everything else',
                '👋 Hand-pick and relocate to wild plants (if you want to save them)',
                '🕸️ Look for and preserve beneficial parasitic wasp cocoons'
            ],
            'natural': [
                '🐦 Encourage birds with bird houses and water sources',
                '🌸 Plant flowers that attract parasitic wasps',
                '🔍 Check plants daily for early detection'
            ]
        },
        'when_to_treat': 'Treat when caterpillars are small (easier to control)',
        'safety_notes': '🌟 Bt is completely safe for humans, pets, and beneficial insects',
        'success_tips': 'Identify the caterpillar first - some become beautiful beneficial butterflies!'
    },
    
    'Earthworms': {
        'description': '🪱 Segmented soil-dwelling decomposers that are absolutely essential for healthy soil.',
        'impact': '🌟 **GARDEN HEROES**: Aerate soil, create nutrient-rich castings, and improve soil structure.',
        'severity': 'EXTREMELY BENEFICIAL',
        'beneficial': True,
        'treatments': {
            'support': [
                '🍂 Add organic matter like compost and leaf mulch',
                '💧 Keep soil consistently moist but not waterlogged',
                '🚫 NEVER use chemical fertilizers or pesticides',
                '🌱 Minimize soil disturbance and digging'
            ]
        },
        'when_to_treat': 'NEVER TREAT - Protect at all costs!',
        'safety_notes': '🛡️ Earthworms indicate healthy soil - their presence is a great sign',
        'success_tips': 'More earthworms = healthier, more productive soil!'
    },
    
    'Earwigs': {
        'description': '🦂 Brown insects with prominent pincers (forceps) at their rear end, active at night.',
        'impact': '⚖️ **Mixed Impact**: May eat seedlings and soft fruits, but also consume many harmful pests.',
        'severity': 'Low',
        'beneficial': 'Mixed',
        'treatments': {
            'trapping': [
                '📰 Roll up damp newspaper, place near plants overnight, shake out in morning',
                '🥫 Set up cardboard tube traps filled with straw',
                '🍺 Create shallow beer traps (they\'re attracted to yeast)'
            ],
            'habitat_modification': [
                '🌞 Keep mulch dry and away from plant stems',
                '🧹 Remove hiding places like boards and debris',
                '💧 Water plants in morning so soil is drier at night'
            ]
        },
        'when_to_treat': 'Only if they\'re causing visible damage - often they\'re helping by eating pests',
        'safety_notes': '✅ These methods don\'t harm other beneficial insects',
        'success_tips': 'Monitor for a week before treating - they might be helping more than harming'
    },
    
    'Grasshoppers': {
        'description': '🦗 Jumping insects with powerful hind legs and strong chewing mouthparts.',
        'impact': '🚨 **High Threat**: Can consume large amounts of foliage quickly, especially in groups.',
        'severity': 'High',
        'beneficial': False,
        'treatments': {
            'physical': [
                '🛡️ Cover seedlings with lightweight row covers',
                '🕳️ Create trenches around garden beds (they can\'t jump out)',
                '🧤 Hand-pick early in morning when they\'re sluggish'
            ],
            'natural_predators': [
                '🐓 Encourage chickens or guinea fowl if you have space',
                '🐦 Attract birds with feeders and water sources',
                '🕷️ Preserve spider habitats'
            ],
            'repellent': [
                '🧄 Garlic spray: blend 3 cloves + 1 tsp soap + 1 quart water',
                '🌶️ Hot pepper spray mixed with a few drops of dish soap'
            ]
        },
        'when_to_treat': 'Prevention is key - treat at first sign before populations explode',
        'safety_notes': '✅ All methods are safe for humans and beneficial insects',
        'success_tips': 'Grasshopper swarms are hard to control - focus on prevention and barriers'
    },
    
    'Moths': {
        'description': '🦋 Nocturnal flying insects; while adults may seem harmless, they lay eggs that become destructive caterpillars.',
        'impact': '⚠️ **Indirect Threat**: Adults lay eggs that hatch into crop-damaging caterpillars.',
        'severity': 'Medium',
        'beneficial': False,
        'treatments': {
            'disruption': [
                '💕 Use pheromone traps to disrupt mating cycles',
                '💡 Set up light traps away from crops to draw them away',
                '🌙 Time outdoor lighting to minimize attraction'
            ],
            'natural_control': [
                '🦇 Encourage bats with bat houses (they eat thousands of insects nightly)',
                '🕷️ Preserve spider webs and habitats',
                '🌸 Plant flowers that attract parasitic wasps'
            ]
        },
        'when_to_treat': 'Monitor closely during warm seasons when they\'re most active',
        'safety_notes': '🌟 Focus on prevention rather than direct killing',
        'success_tips': 'Adult moth control prevents the next generation of caterpillar damage'
    },
    
    'Slugs': {
        'description': '🐌 Soft-bodied mollusks without shells that love moisture and feed at night.',
        'impact': '🚨 **Moderate Threat**: Chew irregular holes in leaves, fruits, and can destroy seedlings overnight.',
        'severity': 'Medium',
        'beneficial': False,
        'treatments': {
            'barriers': [
                '🥉 Create copper tape barriers around raised beds',
                '🥚 Sprinkle crushed eggshells around plants',
                '🧂 Use food-grade diatomaceous earth (reapply after rain)'
            ],
            'traps': [
                '🍺 Set shallow beer traps (they drown in it)',
                '🥬 Use grapefruit rinds as hiding spots, collect in morning',
                '📋 Place boards near plants, flip and collect slugs underneath'
            ],
            'environmental': [
                '🌅 Water plants in early morning instead of evening',
                '🌞 Improve drainage to reduce moisture',
                '🧹 Remove hiding places like debris and dense vegetation'
            ]
        },
        'when_to_treat': 'Best treated at night when they\'re active, or early morning',
        'safety_notes': '⚠️ Iron phosphate pellets are safer than metaldehyde if you choose commercial baits',
        'success_tips': 'Reducing moisture at night dramatically reduces slug problems'
    },
    
    'Snails': {
        'description': '🐌 Similar to slugs but with protective shells, also moisture-loving and nocturnal.',
        'impact': '🚨 **Moderate Threat**: Feed on leafy greens, fruits, and flowers, leaving slime trails.',
        'severity': 'Medium',
        'beneficial': False,
        'treatments': {
            'physical_removal': [
                '👋 Hand-pick in early morning or evening with flashlight',
                '🏺 Drop collected snails in soapy water',
                '🚚 Relocate to wild areas away from garden'
            ],
            'barriers': [
                '🥚 Create barriers with crushed eggshells',
                '🏖️ Use coarse sand around plant bases',
                '🥉 Install copper strips around garden beds'
            ],
            'habitat_modification': [
                '🧹 Remove debris, boards, and hiding spots',
                '✂️ Trim vegetation to reduce cool, moist hiding places',
                '🌞 Improve air circulation around plants'
            ]
        },
        'when_to_treat': 'Evening treatment is most effective when snails are active',
        'safety_notes': '✅ Hand-picking and barriers are completely safe methods',
        'success_tips': 'Consistency in habitat modification prevents re-infestation'
    },
    
    'Wasps': {
        'description': '🐝 Predatory flying insects with narrow waists that hunt other insects for food.',
        'impact': '🌟 **BENEFICIAL PREDATORS**: Eat many harmful garden pests including caterpillars and aphids.',
        'severity': 'BENEFICIAL',
        'beneficial': True,
        'treatments': {
            'coexistence': [
                '🏡 Only remove nests if they pose immediate threat to human safety',
                '👥 Contact professional pest control for nest removal if necessary',
                '🌸 Provide flowering plants to support beneficial wasps',
                '💧 Offer shallow water sources away from high-traffic areas'
            ]
        },
        'when_to_treat': 'Generally avoid treatment - they\'re valuable pest controllers',
        'safety_notes': '⚠️ If allergic to stings, maintain respectful distance but don\'t eliminate them',
        'success_tips': 'Wasps are your garden\'s natural pest control team!'
    },
    
    'Weevils': {
        'description': '🪲 Small beetles with distinctive elongated snouts that damage roots, fruits, and stored grains.',
        'impact': '🚨 **Moderate to High Threat**: Can damage plant roots, bore into fruits, and ruin stored harvests.',
        'severity': 'Medium to High',
        'beneficial': False,
        'treatments': {
            'biological': [
                '🪱 Apply beneficial nematodes to soil (natural weevil predators)',
                '🕷️ Encourage ground beetles and spiders',
                '🐦 Attract birds that eat ground-dwelling insects'
            ],
            'physical': [
                '🟡 Use yellow sticky traps for flying adult weevils',
                '🛡️ Cover plants with row covers during egg-laying season',
                '👋 Hand-pick adult weevils in early morning'
            ],
            'cultural': [
                '🔄 Rotate crops annually to break life cycles',
                '🧹 Remove and destroy plant debris after harvest',
                '❄️ Till soil in fall to expose overwintering larvae to cold'
            ]
        },
        'when_to_treat': 'Start treatment early in growing season before populations establish',
        'safety_notes': '✅ Beneficial nematodes and cultural controls are completely safe',
        'success_tips': 'Prevention through crop rotation is more effective than reactive treatment'
    }
}

# Enhanced prediction display function
def format_treatment_info(pest_name, treatment_data):
    """Format treatment information in a human-readable way"""
    
    if not treatment_data:
        return "No treatment information available for this pest."
    
    # Determine emoji based on beneficial status
    if treatment_data.get('beneficial') == True:
        header_emoji = "🌟"
        action_word = "PROTECT"
    elif treatment_data.get('beneficial') == 'Mixed':
        header_emoji = "⚖️"
        action_word = "MONITOR"
    else:
        header_emoji = "🎯"
        action_word = "MANAGE"
    
    # Build formatted response
    response = f"## {header_emoji} {pest_name} - {action_word}\n\n"
    
    # Description
    response += f"**What it is:** {treatment_data.get('description', 'No description available')}\n\n"
    
    # Impact
    response += f"**Garden Impact:** {treatment_data.get('impact', 'Impact unknown')}\n\n"
    
    # Severity indicator
    severity = treatment_data.get('severity', 'Unknown')
    if severity == 'BENEFICIAL' or severity == 'EXTREMELY BENEFICIAL':
        response += f"**🌟 Status:** {severity} - Please protect!\n\n"
    else:
        response += f"**⚠️ Threat Level:** {severity}\n\n"
    
    # Treatment actions
    treatments = treatment_data.get('treatments', {})
    if treatments:
        response += "**🛠️ Recommended Actions:**\n\n"
        
        for category, actions in treatments.items():
            if actions:  # Only show categories that have actions
                category_name = category.replace('_', ' ').title()
                response += f"**{category_name}:**\n"
                for action in actions:
                    response += f"• {action}\n"
                response += "\n"
    
    # When to treat
    when_to_treat = treatment_data.get('when_to_treat')
    if when_to_treat:
        response += f"**⏰ When to Act:** {when_to_treat}\n\n"
    
    # Safety notes
    safety_notes = treatment_data.get('safety_notes')
    if safety_notes:
        response += f"**🛡️ Safety:** {safety_notes}\n\n"
    
    # Success tips
    success_tips = treatment_data.get('success_tips')
    if success_tips:
        response += f"**💡 Pro Tip:** {success_tips}\n\n"
    
    return response

# Update the predict function in CustomPestApp class
class CustomPestApp:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.class_names = ['Ants', 'Bees', 'Beetles', 'Caterpillars', 'Earthworms', 
                           'Earwigs', 'Grasshoppers', 'Moths', 'Slugs', 'Snails', 
                           'Wasps', 'Weevils']
    
    def predict(self, image):
        """Enhanced prediction with better formatting"""
        if image is None:
            return "Please upload an image", ""
        
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
        
        # Enhanced results with confidence indicators
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
        for i, idx in enumerate(top_3_idx, 1):
            pest = self.class_names[idx]
            conf = predictions[0][idx]
            result += f"{i}. **{pest}**: {conf:.1%}\n"
        
        # Get detailed treatment info
        treatment_info = format_treatment_info(predicted_pest, TREATMENTS.get(predicted_pest, {}))
        
        return result, treatment_info
    
    def create_interface(self):
        """Enhanced Gradio interface with better styling - Fixed white page issue"""
        
        with gr.Blocks(
            title="🐛 Smart Garden Pest Identifier",
            theme=gr.themes.Default()  # Use default theme instead of custom CSS
        ) as app:
            
            gr.Markdown("""
            # 🐛 Smart Garden Pest Identifier
            ## 🧠 AI-Powered Pest Recognition & Organic Treatment Advisor
            
            **Simply upload a photo of any garden pest to instantly get:**
            - 🎯 Accurate pest identification
            - 🌿 Safe, organic treatment options
            - ⏰ Best timing for treatment
            - 🛡️ Safety information for you and beneficial insects
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="📸 Upload Pest Photo", 
                        type="pil"
                    )
                    
                    identify_btn = gr.Button(
                        "🔍 Identify Pest & Get Treatment Advice", 
                        variant="primary"
                    )
                    
                    gr.Markdown("""
                    **📋 Tips for Best Results:**
                    - 📷 Take clear, close-up photos
                    - 🔆 Use good lighting
                    - 🎯 Center the pest in the image
                    - 📏 Include size reference if possible
                    """)
                
                with gr.Column(scale=2):
                    result_output = gr.Markdown(
                        value="Upload an image to see identification results here..."
                    )
                    
                    treatment_output = gr.Markdown(
                        value="Treatment recommendations will appear here after identification..."
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
            
            **🌟 Beneficial Insects:** Bees • Earthworms • Wasps *(These are garden helpers - we'll tell you how to protect them!)*
            
            ## 🧠 Technology Behind This Tool
            - **AI Model:** MobileNetV2 with Transfer Learning
            - **Training:** ImageNet pre-trained (1.4M images) + Custom pest dataset
            - **Accuracy:** Optimized for garden pest identification
            - **Focus:** Organic, safe treatment methods only
            
            ---
            *🌿 Remember: Always try the gentlest treatment methods first, and never harm beneficial insects!*
            """)
        
        return app

# 4. SIMPLE USAGE FUNCTIONS

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

def launch_mobilenetv2_app():
    """Launch MobileNetV2 app"""
    
    if not os.path.exists('mobilenetv2_pest_model.keras'):
        print("❌ No MobileNetV2 model found! Run train_mobilenetv2() first.")
        return
    
    print("🚀 Launching MobileNetV2 Pest App...")
    print("   🧠 Using MobileNetV2 + ImageNet transfer learning")
    print("   🌐 App will open in your default browser")
    print("   📱 Upload pest images to test!")
    
    app = CustomPestApp('mobilenetv2_pest_model.keras')
    interface = app.create_interface()
    interface.launch(share=True)

def run_mobilenetv2_pipeline(force_retrain=False):
    """Smart pipeline for MobileNetV2 with train-val-test evaluation"""
    
    print("🎯 MobileNetV2 Pipeline: Check Model → Train-Val-Test → Launch App")
    
    # Check if model already exists
    if os.path.exists('mobilenetv2_pest_model.keras') and not force_retrain:
        # Get model info
        model_size = os.path.getsize('mobilenetv2_pest_model.keras') / (1024*1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime('mobilenetv2_pest_model.keras'))
        
        print(f"✅ Found existing MobileNetV2 model 'mobilenetv2_pest_model.keras'")
        print(f"   🧠 Type: MobileNetV2 Transfer Learning")
        print(f"   📊 Size: {model_size:.1f}MB")
        print(f"   📅 Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("⏭️ Skipping training - launching app directly...")
        print("   💡 Tip: Use force_retrain=True to retrain anyway")
        
        launch_mobilenetv2_app()
    else:
        if force_retrain and os.path.exists('mobilenetv2_pest_model.keras'):
            print("🔄 Force retrain requested - training new MobileNetV2 model...")
        else:
            print("❌ No existing MobileNetV2 model found")
        
        try:
            history, test_accuracy = train_mobilenetv2()
            print(f"\n🎉 Training complete! Final test accuracy: {test_accuracy*100:.1f}%")
            print("🚀 Launching app...")
            launch_mobilenetv2_app()
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("💡 Make sure you have scikit-learn installed: pip install scikit-learn")

# MOBILENETV2 USAGE
if __name__ == "__main__":
    run_mobilenetv2_pipeline(force_retrain=True)
    # launch_mobilenetv2_app()                         # Just launch app
    # train_mobilenetv2()                              # Just train model