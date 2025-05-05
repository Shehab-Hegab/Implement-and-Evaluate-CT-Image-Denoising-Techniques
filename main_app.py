import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QSplitter, QGroupBox, QFormLayout,
    QLineEdit, QStatusBar, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QTextEdit
)
from PyQt6.QtGui import QPixmap, QImage, QBrush, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback
import image_processor

class FilterWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, filter_type, input_slice, param=None):
        super().__init__()
        self.filter_type = filter_type
        self.input_slice = input_slice
        self.param = param
        
    def run(self):
        try:
            self.progress.emit(f"Processing {self.filter_type} filter...")

            result = None
            if self.filter_type == "Gaussian":
                result = image_processor.apply_gaussian_filter(
                    self.input_slice, 
                    sigma=self.param if self.param is not None else 1.0
                )
            elif self.filter_type == "Bilateral":
                result = image_processor.apply_bilateral_filter(
                    self.input_slice, 
                    sigma_spatial=self.param if self.param is not None else 1.0, 
                    sigma_color=0.1
                )
            elif self.filter_type == "Wavelet":
                result = image_processor.apply_wavelet_filter(self.input_slice)
            elif self.filter_type == "Deep Learning":
                result = image_processor.apply_deep_learning_filter(self.input_slice)
            else:
                self.error.emit(f"Unknown filter type: {self.filter_type}")
                return

            self.finished.emit(result)
            
        except Exception as e:
            traceback_str = traceback.format_exc()
            self.error.emit(f"Error in {self.filter_type} filter: {str(e)}\n{traceback_str}")

class CTImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.original_data = None
        self.original_slice = None
        self.noisy_slice = None
        self.processed_slice = None
        self.current_slice_index = 0
        self.current_axis = 2  # Default to axial view (0=sagittal, 1=coronal, 2=axial)
        self.denoised_results = {}  # Dictionary to store results from different methods
        self.filter_worker = None  # Will hold the worker thread
        
        # Patient info default values
        self.patient_age = "55"
        self.patient_gender = "Male"
        self.patient_clinical_note = "Suspected pulmonary embolism. Focus on left lung."
        
        self.setWindowTitle("CT Image Processor")
        self.resize(1200, 800)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setVisible(False)  # Hide initially
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        self.setup_ui()
        self.status_bar.showMessage("Ready. Load a CT image to begin.")

    def setup_ui(self):
        """Set up the user interface components."""
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        
        # Create splitter for resizable panels
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.splitter.addWidget(left_panel)
        
        # File controls
        file_group = QGroupBox("File Controls")
        # In the setup_ui method, add a save results button to the file controls group
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)
        
        self.load_button = QPushButton("Load CT Image")
        self.load_button.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_button)
        
        # Add the save results button
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)  # Disable until we have results to save
        file_layout.addWidget(self.save_button)
        
        # Slice controls
        slice_group = QGroupBox("Slice Controls")
        slice_layout = QFormLayout()
        slice_group.setLayout(slice_layout)
        
        self.slice_slider = QLineEdit("0")
        slice_layout.addRow("Slice Index:", self.slice_slider)
        
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Axial", "Coronal", "Sagittal"])
        slice_layout.addRow("View:", self.view_combo)
        
        self.update_slice_button = QPushButton("Update Slice")
        self.update_slice_button.clicked.connect(self.update_slice)
        slice_layout.addWidget(self.update_slice_button)
        
        # Noise controls
        noise_group = QGroupBox("Noise Controls")
        noise_layout = QVBoxLayout()
        noise_group.setLayout(noise_layout)
        
        self.noise_type_combo = QComboBox()
        self.noise_type_combo.addItems(["Gaussian", "Salt & Pepper"])
        noise_layout.addWidget(self.noise_type_combo)
        
        noise_param_layout = QFormLayout()
        self.noise_param_input = QLineEdit("25")
        noise_param_layout.addRow("Noise Level:", self.noise_param_input)
        noise_layout.addLayout(noise_param_layout)
        
        self.add_noise_button = QPushButton("Add Noise")
        self.add_noise_button.clicked.connect(self.add_noise)
        noise_layout.addWidget(self.add_noise_button)
        
        # Filter controls
        filter_group = QGroupBox("Filter Controls")
        filter_layout = QVBoxLayout()
        filter_group.setLayout(filter_layout)
        
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["Gaussian", "Bilateral", "Wavelet", "Deep Learning"])
        filter_layout.addWidget(self.filter_type_combo)
        
        filter_param_layout = QFormLayout()
        self.filter_param_input = QLineEdit("1.0")
        filter_param_layout.addRow("Parameter:", self.filter_param_input)
        filter_layout.addLayout(filter_param_layout)
        
        self.apply_filter_button = QPushButton("Apply Filter")
        self.apply_filter_button.clicked.connect(self.apply_filter)
        filter_layout.addWidget(self.apply_filter_button)
        
        self.cancel_button = QPushButton("Cancel Processing")
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setVisible(False)
        filter_layout.addWidget(self.cancel_button)
        
        self.compare_all_button = QPushButton("Compare All Methods")
        self.compare_all_button.clicked.connect(self.compare_all_methods)
        filter_layout.addWidget(self.compare_all_button)
        
        # Add all control groups to left panel
        left_layout.addWidget(file_group)
        left_layout.addWidget(slice_group)
        left_layout.addWidget(noise_group)
        left_layout.addWidget(filter_group)
        left_layout.addStretch()
        
        # Right panel - Image display and results
        right_panel = QTabWidget()
        self.splitter.addWidget(right_panel)
        
        # Images tab
        images_widget = QWidget()
        images_layout = QVBoxLayout(images_widget)
        
        # Image displays
        image_displays = QWidget()
        image_displays_layout = QHBoxLayout(image_displays)
        
        # Original image
        original_group = QGroupBox("Original")
        original_layout = QVBoxLayout()
        original_group.setLayout(original_layout)
        self.original_scene = QGraphicsScene()
        self.original_view = QGraphicsView(self.original_scene)
        self.original_view.setMinimumSize(300, 300)
        original_layout.addWidget(self.original_view)
        image_displays_layout.addWidget(original_group)
        
        # Noisy image
        noisy_group = QGroupBox("Noisy")
        noisy_layout = QVBoxLayout()
        noisy_group.setLayout(noisy_layout)
        self.noisy_scene = QGraphicsScene()
        self.noisy_view = QGraphicsView(self.noisy_scene)
        self.noisy_view.setMinimumSize(300, 300)
        noisy_layout.addWidget(self.noisy_view)
        image_displays_layout.addWidget(noisy_group)
        
        # Processed image
        processed_group = QGroupBox("Processed")
        processed_layout = QVBoxLayout()
        processed_group.setLayout(processed_layout)
        self.processed_scene = QGraphicsScene()
        self.processed_view = QGraphicsView(self.processed_scene)
        self.processed_view.setMinimumSize(300, 300)
        processed_layout.addWidget(self.processed_view)
        image_displays_layout.addWidget(processed_group)
        
        images_layout.addWidget(image_displays)
        
        # Bottom panel with info and medical data
        bottom_panel = QWidget()
        bottom_layout = QHBoxLayout(bottom_panel)
        
        # Info panel
        self.info_panel = QGroupBox("Information")
        info_layout = QFormLayout()
        self.info_panel.setLayout(info_layout)
        
        self.psnr_label = QLabel("N/A")
        info_layout.addRow("PSNR:", self.psnr_label)
        
        self.ssim_label = QLabel("N/A")
        info_layout.addRow("SSIM:", self.ssim_label)
        
        self.filter_info_label = QLabel("No filter applied")
        info_layout.addRow("Filter:", self.filter_info_label)
        
        bottom_layout.addWidget(self.info_panel)
        
        # Medical data panel (right side)
        medical_panel = QWidget()
        medical_layout = QVBoxLayout(medical_panel)
        
        # 1. Patient Info Panel
        patient_group = QGroupBox("Patient Info")
        patient_layout = QFormLayout()
        patient_group.setLayout(patient_layout)
        
        self.patient_age_label = QLabel(self.patient_age)
        patient_layout.addRow("Age:", self.patient_age_label)
        
        self.patient_gender_label = QLabel(self.patient_gender)
        patient_layout.addRow("Gender:", self.patient_gender_label)
        
        # In the setup_ui method, change the patient_note_text from read-only to editable
        self.patient_note_text = QTextEdit()
        self.patient_note_text.setPlainText(self.patient_clinical_note)
        self.patient_note_text.setMaximumHeight(60)
        self.patient_note_text.textChanged.connect(self.update_clinical_note)
        patient_layout.addRow("Clinical Note:", self.patient_note_text)
        
        medical_layout.addWidget(patient_group)
        
        # 2. System Suggestions
        suggestions_group = QGroupBox("System Suggestions")
        suggestions_layout = QVBoxLayout()
        suggestions_group.setLayout(suggestions_layout)
        
        self.suggestions_label = QLabel("No processing performed yet.")
        self.suggestions_label.setWordWrap(True)
        suggestions_layout.addWidget(self.suggestions_label)
        
        medical_layout.addWidget(suggestions_group)
        
        # 3. Diagnostic Aid Score
        score_group = QGroupBox("Diagnostic Aid Score")
        score_layout = QVBoxLayout()
        score_group.setLayout(score_layout)
        
        score_display = QHBoxLayout()
        
        self.score_value_label = QLabel("N/A")
        self.score_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.score_value_label.setFont(font)
        score_display.addWidget(self.score_value_label)
        
        self.score_text_label = QLabel("")
        self.score_text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_display.addWidget(self.score_text_label)
        
        score_layout.addLayout(score_display)
        
        medical_layout.addWidget(score_group)
        
        bottom_layout.addWidget(medical_panel)
        
        # Add bottom panel to main layout
        images_layout.addWidget(bottom_panel)
        
        # Comparison tab
        comparison_widget = QWidget()
        comparison_layout = QVBoxLayout(comparison_widget)
        
        self.comparison_table = QTableWidget(0, 3)  # Rows will be added dynamically
        self.comparison_table.setHorizontalHeaderLabels(["Method", "PSNR", "SSIM"])
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        comparison_layout.addWidget(self.comparison_table)
        
        # Add tabs to right panel
        right_panel.addTab(images_widget, "Images")
        right_panel.addTab(comparison_widget, "Comparison")
        
        # Set initial splitter sizes
        self.splitter.setSizes([300, 900])
        
        # Disable buttons until image is loaded
        self.update_slice_button.setEnabled(False)
        self.add_noise_button.setEnabled(False)
        self.apply_filter_button.setEnabled(False)
        self.compare_all_button.setEnabled(False)

    def load_image(self):
        """Load a CT image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CT Image", "", "NIfTI Files (*.nii *.nii.gz);;All Files (*)"
        )
        
        if file_path:
            self.status_bar.showMessage(f"Loading {file_path}...")
            self.original_data = image_processor.load_nifti_file(file_path)
            
            if self.original_data is not None:
                self.current_slice_index = self.original_data.shape[2] // 2  # Middle slice for axial view
                self.slice_slider.setText(str(self.current_slice_index))
                self.update_slice()
                self.status_bar.showMessage(f"Loaded {file_path}")
                
                # Enable controls
                self.update_slice_button.setEnabled(True)
                self.add_noise_button.setEnabled(True)
            else:
                self.status_bar.showMessage("Failed to load image")

    def update_slice(self):
        """Update the displayed slice based on current settings."""
        if self.original_data is None:
            return
            
        try:
            slice_index = int(self.slice_slider.text())
            view_index = self.view_combo.currentIndex()
            self.current_axis = 2 - view_index  # Convert from UI index to axis (2=axial, 1=coronal, 0=sagittal)
            
            # Get the slice
            self.original_slice = image_processor.get_slice(self.original_data, slice_index, self.current_axis)
            
            if self.original_slice is None:
                self.status_bar.showMessage("Invalid slice index")
                return
                
            # Normalize for display
            self.original_slice = image_processor.normalize_image(self.original_slice)
            
            # Clear previous results when slice changes
            self.noisy_slice = None
            self.processed_slice = None
            self.denoised_results.clear()
            
            # Update displays
            self.update_image_displays()
            self.update_comparison_table()
            
            # Enable/disable buttons
            self.add_noise_button.setEnabled(True)
            self.apply_filter_button.setEnabled(False)
            self.compare_all_button.setEnabled(False)
            
            self.status_bar.showMessage(f"Updated to {self.view_combo.currentText()} view, slice {slice_index}")
            
        except ValueError:
            self.status_bar.showMessage("Invalid slice index")

    def add_noise(self):
        """Add noise to the current slice."""
        if self.original_slice is None:
            return
            
        noise_type = self.noise_type_combo.currentText()
        
        try:
            param = float(self.noise_param_input.text())
            if param < 0:
                raise ValueError("Parameter cannot be negative")
                
            if noise_type == "Gaussian":
                self.noisy_slice = image_processor.add_gaussian_noise_func(self.original_slice, sigma=param)
            else:  # Salt & Pepper
                self.noisy_slice = image_processor.add_salt_pepper_noise(self.original_slice, amount=param/100)
                
            self.update_image_displays()
            self.apply_filter_button.setEnabled(True)
            self.status_bar.showMessage(f"Added {noise_type} noise")
            
        except ValueError as e:
            self.status_bar.showMessage(f"Invalid parameter: {e}")

    def update_image_displays(self):
        """Update all image displays with current data."""
        # Clear scenes
        self.original_scene.clear()
        self.noisy_scene.clear()
        self.processed_scene.clear()
        
        # Update original image
        if self.original_slice is not None:
            original_qimage = image_processor.convert_to_qimage(self.original_slice)
            if original_qimage:
                self.original_scene.addPixmap(QPixmap.fromImage(original_qimage))
                self.original_view.fitInView(self.original_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        
        # Update noisy image
        if self.noisy_slice is not None:
            noisy_qimage = image_processor.convert_to_qimage(self.noisy_slice)
            if noisy_qimage:
                self.noisy_scene.addPixmap(QPixmap.fromImage(noisy_qimage))
                self.noisy_view.fitInView(self.noisy_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        
        # Update processed image
        if self.processed_slice is not None:
            processed_qimage = image_processor.convert_to_qimage(self.processed_slice)
            if processed_qimage:
                self.processed_scene.addPixmap(QPixmap.fromImage(processed_qimage))
                self.processed_view.fitInView(self.processed_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def update_info_panel(self, filter_type=None, filter_param=None):
        """Update the information panel with metrics."""
        if self.original_slice is not None and self.processed_slice is not None:
            # Calculate metrics
            psnr_val, ssim_val = image_processor.calculate_metrics(
                self.original_slice, self.processed_slice
            )
            
            if psnr_val is not None:
                self.psnr_label.setText(f"{psnr_val:.2f} dB")
            else:
                self.psnr_label.setText("N/A")
                
            if ssim_val is not None:
                self.ssim_label.setText(f"{ssim_val:.4f}")
            else:
                self.ssim_label.setText("N/A")
                
            if filter_type:
                param_text = f", param={filter_param:.2f}" if filter_param is not None else ""
                self.filter_info_label.setText(f"{filter_type}{param_text}")
                
            # Update diagnostic aid score
            self.update_diagnostic_score(psnr_val, ssim_val)
            
            # Update system suggestions
            self.update_system_suggestions(filter_type, psnr_val, ssim_val)
        else:
            self.psnr_label.setText("N/A")
            self.ssim_label.setText("N/A")
            self.filter_info_label.setText("No filter applied")
            self.score_value_label.setText("N/A")
            self.score_text_label.setText("")
            self.suggestions_label.setText("No processing performed yet.")

    def update_diagnostic_score(self, psnr_val, ssim_val):
        """Calculate and update the diagnostic aid score based on PSNR and SSIM values."""
        if psnr_val is None or ssim_val is None:
            self.score_value_label.setText("N/A")
            self.score_text_label.setText("")
            return
            
        # Calculate score based on PSNR and SSIM
        score = 1  # Default lowest score
        
        if ssim_val > 0.6 and psnr_val > 25:
            score = 5
            quality = "Excellent"
        elif ssim_val > 0.5 and psnr_val > 22:
            score = 4
            quality = "Good"
        elif ssim_val > 0.4 and psnr_val > 18:
            score = 3
            quality = "Acceptable"
        elif ssim_val > 0.2:
            score = 2
            quality = "Poor"
        else:
            score = 1
            quality = "Very Poor"
            
        self.score_value_label.setText(str(score))
        self.score_text_label.setText(f"({quality})")
        
        # Set color based on score
        if score >= 4:
            self.score_value_label.setStyleSheet("color: green;")
        elif score == 3:
            self.score_value_label.setStyleSheet("color: orange;")
        else:
            self.score_value_label.setStyleSheet("color: red;")

    def update_system_suggestions(self, filter_type, psnr_val, ssim_val):
        """Update system suggestions based on processing results."""
        suggestions = []
        
        # Check noise level
        try:
            noise_level = float(self.noise_param_input.text())
            if noise_level > 30:
                suggestions.append("High noise level detected. Consider re-acquiring image or applying stronger denoising.")
        except (ValueError, AttributeError):
            pass
            
        # Check filter performance
        if filter_type and psnr_val is not None and ssim_val is not None:
            if psnr_val < 15:
                suggestions.append(f"{filter_type} filter performed poorly. Try a different denoising method.")
            elif psnr_val > 25 and ssim_val > 0.6:
                suggestions.append(f"{filter_type} filtering performed excellently. Recommended for diagnostic use.")
                
        # Check best method from comparison
        best_method = self.find_best_method()
        if best_method and best_method != filter_type:
            suggestions.append(f"{best_method} filtering performed best in previous tests. Consider using it instead.")
            
        # Set the suggestions text
        if suggestions:
            self.suggestions_label.setText("\n".join(suggestions))
        else:
            self.suggestions_label.setText("No specific suggestions available.")

    def find_best_method(self):
        """Find the best performing method from the comparison results."""
        if not self.denoised_results:
            return None
            
        best_method = None
        best_score = -1
        
        for method in self.denoised_results:
            result = self.denoised_results[method]
            psnr, ssim = image_processor.calculate_metrics(self.original_slice, result)
            
            if psnr is not None and ssim is not None:
                # Simple combined score
                score = psnr * ssim
                if score > best_score:
                    best_score = score
                    best_method = method
                    
        return best_method

    def update_comparison_table(self, just_applied_method=None):
        """Update the comparison table with results from all methods."""
        self.comparison_table.setRowCount(0)  # Clear table
        
        if not self.denoised_results or self.original_slice is None:
            return
            
        # Add rows for each method
        for i, (method, result) in enumerate(self.denoised_results.items()):
            self.comparison_table.insertRow(i)
            
            # Method name
            method_item = QTableWidgetItem(method)
            if method == just_applied_method:
                method_item.setBackground(QBrush(Qt.GlobalColor.yellow))
            self.comparison_table.setItem(i, 0, method_item)
            
            # Calculate metrics
            psnr_val, ssim_val = image_processor.calculate_metrics(
                self.original_slice, result
            )
            
            # PSNR
            psnr_item = QTableWidgetItem(f"{psnr_val:.2f} dB" if psnr_val is not None else "N/A")
            self.comparison_table.setItem(i, 1, psnr_item)
            
            # SSIM
            ssim_item = QTableWidgetItem(f"{ssim_val:.4f}" if ssim_val is not None else "N/A")
            self.comparison_table.setItem(i, 2, ssim_item)

    def compare_all_methods(self):
        """Run all denoising methods and compare results."""
        if self.noisy_slice is None and self.original_slice is None:
            self.status_bar.showMessage("Load an image and apply noise first")
            return
            
        input_slice = self.noisy_slice if self.noisy_slice is not None else self.original_slice
        
        # Disable UI during processing
        self.apply_filter_button.setEnabled(False)
        self.add_noise_button.setEnabled(False)
        self.compare_all_button.setEnabled(False)
        
        # Process with each method
        methods = ["Gaussian", "Bilateral", "Wavelet", "Deep Learning"]
        
        for method in methods:
            if method in self.denoised_results:
                continue  # Skip if already processed
                
            self.status_bar.showMessage(f"Processing {method}...")
            
            # Create and run worker for this method
            param = 1.0 if method in ["Gaussian", "Bilateral"] else None
            worker = FilterWorker(method, input_slice, param)
            
            # We're running synchronously for the comparison
            try:
                result = None
                if method == "Gaussian":
                    result = image_processor.apply_gaussian_filter(input_slice, sigma=param)
                elif method == "Bilateral":
                    result = image_processor.apply_bilateral_filter(input_slice, sigma_spatial=param)
                elif method == "Wavelet":
                    result = image_processor.apply_wavelet_filter(input_slice)
                elif method == "Deep Learning":
                    result = image_processor.apply_deep_learning_filter(input_slice)
                    
                if result is not None:
                    self.denoised_results[method] = result
            except Exception as e:
                print(f"Error processing {method}: {e}")
        
        # Update the table with all results
        self.update_comparison_table()
        
        # Re-enable UI
        self.apply_filter_button.setEnabled(True)
        self.add_noise_button.setEnabled(True)
        self.compare_all_button.setEnabled(True)
        
        self.status_bar.showMessage("Comparison complete")

    def apply_filter(self):
        input_slice = self.noisy_slice if self.noisy_slice is not None else self.original_slice
    
        if input_slice is None:
            self.status_bar.showMessage("Load an image or apply noise first.")
            return
    
        filter_type = self.filter_type_combo.currentText()
        
        param = None
        if filter_type not in ["Wavelet", "Deep Learning"]:
            try:
                param = float(self.filter_param_input.text())
                if param < 0:
                    raise ValueError("Parameter cannot be negative.")
            except ValueError as e:
                self.status_bar.showMessage(f"Invalid parameter value: {e}")
                return
    
        self.apply_filter_button.setEnabled(False)
        self.add_noise_button.setEnabled(False)
        self.compare_all_button.setEnabled(False)
        
        self.status_bar.showMessage(f"Starting {filter_type} filter processing...")
        
        self.filter_worker = FilterWorker(filter_type, input_slice, param)
        
        self.filter_worker.finished.connect(self.on_filter_finished)
        self.filter_worker.error.connect(self.handle_filter_error)
        self.filter_worker.progress.connect(self.status_bar.showMessage)
        
        self.progress_bar.setVisible(True)
        self.cancel_button.setVisible(True)
        
        self.filter_worker.start()

    def on_filter_finished(self, result):
        """Handle when a filter operation is complete."""
        self.processed_slice = result
        self.update_image_displays()
        
        # Store result for comparison
        filter_type = self.filter_type_combo.currentText()
        self.denoised_results[filter_type] = result
        
        # Update info panel
        param = None
        if filter_type not in ["Wavelet", "Deep Learning"]:
            try:
                param = float(self.filter_param_input.text())
            except ValueError:
                pass
        self.update_info_panel(filter_type, param)
        
        # Update comparison table
        self.update_comparison_table(filter_type)
        
        # Re-enable UI
        self.apply_filter_button.setEnabled(True)
        self.add_noise_button.setEnabled(True)
        self.compare_all_button.setEnabled(True)
        self.save_button.setEnabled(True)  # Enable save button
        self.cancel_button.setVisible(False)
        self.progress_bar.setVisible(False)
        
        self.status_bar.showMessage(f"Applied {filter_type} filter")

    def handle_filter_error(self, error_message):
        print(error_message)
        self.status_bar.showMessage(f"Error: {error_message}")
        
        self.apply_filter_button.setEnabled(True)
        self.add_noise_button.setEnabled(True)
        self.compare_all_button.setEnabled(True)
        self.cancel_button.setVisible(False)
        self.progress_bar.setVisible(False)

    def cancel_processing(self):
        if self.filter_worker and self.filter_worker.isRunning():
            self.filter_worker.terminate()
            self.filter_worker.wait()
            self.status_bar.showMessage("Processing cancelled.")
            self.progress_bar.setVisible(False)
            self.cancel_button.setVisible(False)
            
            self.apply_filter_button.setEnabled(True)
            self.add_noise_button.setEnabled(True)
            self.compare_all_button.setEnabled(True)

    def update_clinical_note(self):
        """Update the stored clinical note when the text is changed."""
        self.patient_clinical_note = self.patient_note_text.toPlainText()

    def save_results(self):
        """Save the processed image results to a file."""
        if self.processed_slice is None:
            self.status_bar.showMessage("No processed image to save.")
            return
            
        # Ask user for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if not file_path:
            return  # User cancelled
            
        # Convert the processed image to QImage
        qimage = image_processor.convert_to_qimage(self.processed_slice)
        
        if qimage and qimage.save(file_path):
            self.status_bar.showMessage(f"Image saved to {file_path}")
        else:
            self.status_bar.showMessage("Failed to save image")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = CTImageProcessorApp()
    main_window.show()
    sys.exit(app.exec())