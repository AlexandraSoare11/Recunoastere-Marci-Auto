package com.example.recunoasteremasini;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.DataType;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import android.content.res.AssetFileDescriptor;

public class VCNN extends AppCompatActivity {
    private static final int NUM_CLASSES = 19; // Numărul de clase din modelul tău

    private ImageView imageView;
    private TextView textViewResult;
    private TextView textViewProbabilities; // Adaugă această linie
    private Interpreter tflite;

    private ActivityResultLauncher<Intent> loadImageLauncher;

    private String[] classNames = {
            "Audi", "Bentley", "Benz", "Bmw", "Cadillac",
            "Dodge", "Ferrari", "Ford", "Ford mustang", "Kia",
            "Lamborghini", "Lexus", "Maserati", "Porsche", "Rolls royce",
            "Tesla", "Toyota", "alfa romeo", "hyundai"
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_vcnn);

        imageView = findViewById(R.id.imageView);
        textViewResult = findViewById(R.id.textViewResult);
        textViewProbabilities = findViewById(R.id.textViewProbabilities); // Inițializează aici

        // Initialize ActivityResultLauncher for loading images
        loadImageLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        try {
                            Bitmap imageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), result.getData().getData());

                            // Redimensionare imagine înainte de clasificare
                            Bitmap resizedBitmap = Bitmap.createScaledBitmap(imageBitmap, 128, 128, true);

                            imageView.setImageBitmap(imageBitmap); // Afișare imagine originală
                            classifyImage(resizedBitmap);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }
        );

        findViewById(R.id.button_load_photo).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent loadImageIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                if (loadImageIntent.resolveActivity(getPackageManager()) != null) {
                    loadImageLauncher.launch(loadImageIntent);
                }
            }
        });

        // Load your TensorFlow Lite model
        try {
            tflite = new Interpreter(loadModelFile("mymodelVCNN.tflite"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        Log.d("VCNN", "onPause called");
    }

    @Override
    protected void onStop() {
        super.onStop();
        Log.d("VCNN", "onStop called");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.d("VCNN", "onDestroy called");
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
    }

    private MappedByteBuffer loadModelFile(String modelName) throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(modelName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void classifyImage(Bitmap imageBitmap) {
        long startTime = System.currentTimeMillis();

        // Preprocess the image to the input size of the model
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(imageBitmap);

        // Prepare the input and output buffer
        TensorBuffer inputBuffer = TensorBuffer.createFixedSize(new int[]{1, 128, 128, 3}, DataType.FLOAT32);
        inputBuffer.loadBuffer(tensorImage.getBuffer());

        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(new int[]{1, NUM_CLASSES}, DataType.FLOAT32);

        // Run the inference
        tflite.run(inputBuffer.getBuffer(), outputBuffer.getBuffer().rewind());

        long endTime = System.currentTimeMillis();
        long inferenceTime = endTime - startTime;

        // Process the output
        float[] outputArray = outputBuffer.getFloatArray();
        int maxIndex = 0;
        float maxConfidence = 0;
        for (int i = 0; i < outputArray.length; i++) {
            if (outputArray[i] > maxConfidence) {
                maxConfidence = outputArray[i];
                maxIndex = i;
            }
        }
        StringBuilder probabilitiesText = new StringBuilder("Probabilities:\n");
        for (int i = 0; i < outputArray.length; i++) {
            probabilitiesText.append(classNames[i]).append(": ").append(outputArray[i]).append("\n");
        }

        // Display the result
        String resultText = "Result: " + classNames[maxIndex] + " (Confidence: " + maxConfidence + ") - Time: " + inferenceTime + "ms";
        textViewResult.setText(resultText);
        textViewProbabilities.setText(probabilitiesText.toString());
    }
}
