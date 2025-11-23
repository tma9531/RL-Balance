using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using Random = UnityEngine.Random;

public class SimulatedAgent : Agent
{
    [Header("Graphics Settings (0–1 normalized)")]
    [Range(0, 1)] public float resolutionLevel;
    [Range(0, 1)] public float textureQuality;
    [Range(0, 1)] public float shadowQuality;
    [Range(0, 1)] public float antiAliasing;

    [Header("Performance Measurement")]
    public float simulatedFPS;

    //private const float TARGET_FRAME_WINDOW = 1f; // 1 action per second for presenting
    // Trains much faster, one action each update
    // can use multiple envs, using 8

    private float previousFPS;
    public float quality;
    public int actionCount;
    public float episodeTime = 0f;
    private float maxEpisodeTime = 100f; // 100 actions for training
    //private float maxEpisodeTime = 30f; // 30 actions for presenting
    private float episodeFPSSum = 0f;
    private float episodeQualitySum = 0f;

    public int settingIndex;
    public int direction;
    public int strength;

    #region Camera
    UniversalRenderPipelineAsset urp;
    public Transform cameraTransform;

    Vector3[] positions = {
       new(-92.7068f, -17.76666f, 102.7232f),
       new(-81.82189f, 25.03973f, 1116.169f),
       new(94.29424f, -13.02881f, 443.523f),
       new(-341.7031f, -22.81111f, 726.1611f),
       new(944.523f, 231.0513f, -2555.186f),
       new(-871.3297f, 38.8715f, 74.04516f),
       new(-491.7245f, -28.4068f, 415.1801f),
       new(-58.54248f, 2.775518f, 127.9652f),
       new(-333.3901f, -25.32067f, 292.5073f),
       new(-449.1131f, -24.09342f, 410.1732f)
    };

    Vector3[] rotations = {
       new(-0.859f, -23.309f, 0f),
       new(2.852f, 116.947f, -0.223f),
       new(3.38f, 88.486f, -0.575f),
       new(1.146f, 75.617f, -0.574f),
       new(-23.634f, -245.137f, -1.089f),
       new(14.691f, 49.295f, -1.031f),
       new(-2.324f, 91.882f, -0.999f),
       new(37.032f, 94.766f, -1.249f),
       new(6.9f, 62.282f, 0f),
       new(0.884f, 84.284f, 0f)
    };
    #endregion

    #region Initialization
    public override void Initialize()
    {
        ResetSettings();
        urp = GraphicsSettings.currentRenderPipeline as UniversalRenderPipelineAsset;
    }

    public override void OnEpisodeBegin()
    {
        ResetSettings();
    }

    private void ResetSettings()
    {
        resolutionLevel = Random.Range(0.75f, 1f);
        textureQuality = Random.Range(0.75f, 1f);
        shadowQuality = Random.Range(0.75f, 1f);
        antiAliasing = Random.Range(0.75f, 1f);

        simulatedFPS = 30f;
        previousFPS = 30f;
        actionCount = 0;
        episodeTime = 0f;
        episodeFPSSum = 0f;
        episodeQualitySum = 0f;

        RandomizeCamera();
    }

    private void RandomizeCamera()
    {
        int randomCam = Random.Range(0, 10);

        cameraTransform.position = positions[randomCam];
        cameraTransform.eulerAngles = rotations[randomCam];
    }
    #endregion

    #region Observations
    public override void CollectObservations(VectorSensor sensor)
    {
        // Settings
        sensor.AddObservation(resolutionLevel);
        sensor.AddObservation(textureQuality);
        sensor.AddObservation(shadowQuality);
        sensor.AddObservation(antiAliasing);

        // FPS feedback
        previousFPS = simulatedFPS;
        simulatedFPS = SimulatedFPS(resolutionLevel, textureQuality, shadowQuality, antiAliasing);
        sensor.AddObservation(simulatedFPS / 200f);
    }

    float SimulatedFPS(
        float resolution,       // 0–1
        float textureQuality,   // 0–1
        float shadowQuality,    // 0–1
        float aaLevel           // 0–1
    )
    {
        // --- BASELINE ---
        const float BaseFPS = 80f;

        // --- COST FUNCTIONS ---
        // Resolution affects pixel count quadratically
        float costResolution = 2.5f * (resolution * resolution);
        // Shadow quality: nonlinear, very expensive at high values
        float costShadows = 2.0f * (shadowQuality * shadowQuality);
        // Texture quality: mostly linear influence
        float costTextures = 0.6f * textureQuality;
        // Anti-aliasing: cost increases but slower (root)
        float costAA = 1.2f * Mathf.Sqrt(aaLevel);

        // --- TOTAL GPU COST ---
        float gpuCost = costResolution + costShadows + costTextures + costAA;

        // --- Convert cost to FPS (GPU bound) ---
        float fps = BaseFPS / (1f + gpuCost);

        // noise to make RL more robust
         fps *= UnityEngine.Random.Range(0.98f, 1.02f);

        return fps;
    }

    #endregion

    #region Actions
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var discreteActions = actionBuffers.DiscreteActions;
        settingIndex = discreteActions[0]; // which setting
        direction = discreteActions[1];    // 0 = down, 1 = up
        strength = discreteActions[2];     // how much to change: 0-4 (0 = a little, 4 = a lot)

        float delta = Mathf.Lerp(0.1f, 0.5f, strength) * (direction == 0 ? -1 : 1);

        switch (settingIndex)
        {
            case 0:
                resolutionLevel = Mathf.Clamp01(resolutionLevel + delta);
                break;
            case 1:
                textureQuality = Mathf.Clamp01(textureQuality + delta);
                break;
            case 2:
                shadowQuality = Mathf.Clamp01(shadowQuality + delta);
                break;
            case 3:
                antiAliasing = Mathf.Clamp01(antiAliasing + delta);
                break;
        }
    }
    #endregion

    #region Reward and Update

    private void Update()
    {
        
        RequestDecision();
        actionCount++;

        episodeTime++;
        episodeFPSSum += simulatedFPS;

        float reward = BalancingReward();
        //float reward = FPSReward();

        // Apply
        AddReward(reward);

        // End episode conditions
        if (simulatedFPS > 60f)
        {
            float timeReward = (maxEpisodeTime - episodeTime) / maxEpisodeTime;
            AddReward(timeReward);
            LogEpisodeStats();
            EndEpisode();
        }
        else if (quality < 0.1)
        {
            float lowQualityPenalty = -10f;
            AddReward(lowQualityPenalty);
            LogEpisodeStats();
            EndEpisode();
        }
        else if (episodeTime > maxEpisodeTime)
        {
            LogEpisodeStats();
            EndEpisode();
        }
    }

    private void LogEpisodeStats()
    {
        if (episodeTime > 0)
        {
            float avgFPS = episodeFPSSum / episodeTime;
            float avgQuality = episodeQualitySum / episodeTime;

            var sr = Academy.Instance.StatsRecorder;
            sr.Add("Custom/AvgFPS", avgFPS);
            sr.Add("Custom/AvgQuality", avgQuality);
        }
    }

    private float BalancingReward()
    {
        /// === Smoothed FPS normalization with 60 FPS as max ===
        //float fpsNorm = Mathf.Exp(-(Mathf.Max(0f, 60f - measuredFPS) / 15f));

        // Try to incentivize 60 FPS over quality = 1
        float fpsTarget = 60f;
        float fpsNorm = simulatedFPS / fpsTarget;
        float fpsReward = fpsNorm * fpsNorm;

        // === Nonlinear quality score ===
        // Weighted by how important I find each setting is for visual quality 
        quality = (
            0.4f * resolutionLevel +
            0.3f * textureQuality +
            0.2f * shadowQuality +
            0.1f * antiAliasing);
        episodeQualitySum += quality;
        //float qualityNonlinear = quality * quality;       // squaring means the agent values increases from med to high more
        float qualityNonlinear = Mathf.Sqrt(quality);       // square rooting means the agent values increases from low to med more

        // === Tradeoff coefficient ===
        const float alpha = 0.15f;       // importance of quality
        const float beta = 0.1f;        // importance of increasing FPS this step

        // === Final reward ===
        //float reward = fpsNorm - alpha * (1f - qualityNonlinear);
        float fpsDelta = simulatedFPS - previousFPS;
        float reward = fpsReward + (beta * fpsDelta) - (alpha * (1 - qualityNonlinear));

        return reward;
    }

    private float FPSReward()
    {
        // still here for tracking purposes
        quality = (
            0.4f * resolutionLevel +
            0.3f * textureQuality +
            0.2f * shadowQuality +
            0.1f * antiAliasing);
        episodeQualitySum += quality;

        float fpsTarget = 60f;
        float fpsReward = simulatedFPS / fpsTarget;
        return fpsReward;
    }
    #endregion

    #region Heuristic
    // Baseline: choose a random setting and randomly adjust it
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discrete = actionsOut.DiscreteActions;
        discrete[0] = Random.Range(0, 4); // setting
        discrete[1] = Random.Range(0, 2); // direction
    }
    #endregion
}
