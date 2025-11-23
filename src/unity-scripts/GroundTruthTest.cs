using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class GroundTruthTest : MonoBehaviour
{
    [System.Serializable]
    public struct Preset
    {
        public string name;
        public int resolutionLevel;
        public int vSync; // 0 or 1
        public int aa;    // 0,2,4,8
        public int shadowDistanceInt; // use as float later
        public int textureMipmapLimit; // QualitySettings.globalTextureMipmapLimit
    }

    public List<Preset> presets = new List<Preset>();
    public float settleSeconds = 5f;
    public float sampleSeconds = 20f;
    public int repeats = 5;
    public string outCsvPath = "C:\\Users\\tylea\\TerrainURPRL\\ground_truth_test.csv";

    private void Start()
    {
        // Example presets if none provided
        if (presets.Count == 0)
        {
            presets.Add(new Preset { name = "Low", resolutionLevel = 0, vSync = 0, aa = 0, shadowDistanceInt = 20, textureMipmapLimit = 2 });
            presets.Add(new Preset { name = "High", resolutionLevel = 1, vSync = 0, aa = 8, shadowDistanceInt = 150, textureMipmapLimit = 0 });
        }
        StartCoroutine(RunSensitivityTest());
    }

    IEnumerator RunSensitivityTest()
    {
        // disable vSync global to avoid caps
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = -1;

        // Prepare CSV header
        var lines = new List<string>();
        lines.Add("preset,repeat,avgFPS,stdFPS,minFPS,maxFPS");

        foreach (var preset in presets)
        {
            for (int r = 0; r < repeats; r++)
            {
                // apply settings
                //Screen.SetResolution(preset.width, preset.height, FullScreenMode.Windowed);
                UniversalRenderPipelineAsset urp = GraphicsSettings.currentRenderPipeline as UniversalRenderPipelineAsset;
                urp.renderScale = Mathf.Lerp(0.5f, 2.0f, preset.resolutionLevel); // low = 960x540, high = 3840x2160 
                QualitySettings.antiAliasing = preset.aa;
                QualitySettings.shadowDistance = preset.shadowDistanceInt;
                QualitySettings.globalTextureMipmapLimit = preset.textureMipmapLimit;
                QualitySettings.vSyncCount = preset.vSync;

                Debug.Log($"Applied preset {preset.name} (repeat {r}). Waiting {settleSeconds}s to settle.");
                yield return new WaitForSeconds(settleSeconds);

                // sample FPS for sampleSeconds
                float t0 = Time.realtimeSinceStartup;
                float end = t0 + sampleSeconds;
                var samples = new List<float>();
                while (Time.realtimeSinceStartup < end)
                {
                    samples.Add(1f / Time.unscaledDeltaTime);
                    yield return null;
                }

                float avg = samples.Average();
                float min = samples.Min();
                float max = samples.Max();
                float std = Mathf.Sqrt(samples.Select(s => (s - avg) * (s - avg)).Average());

                Debug.Log($"Preset {preset.name} r{r}: avg {avg:F1}, std {std:F2}, min {min:F1}, max {max:F1}");
                lines.Add($"{preset.name},{r},{avg:F2},{std:F2},{min:F2},{max:F2}");
            }
        }

        // save csv
        var path = Path.Combine(Application.dataPath, outCsvPath);
        File.WriteAllLines(path, lines);
        Debug.Log($"Ground truth test finished. CSV saved to {path}");
    }
}
