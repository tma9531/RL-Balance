using TMPro;
using UnityEngine;

public class FPSDisplayer : MonoBehaviour
{
    public PerformanceAgent agent;   // Reference to agent
    public TextMeshProUGUI fpsText; // Reference to UI Text (on Canvas)

    private float timeAccumulator;
    private const float TARGET_FRAME_WINDOW = 0.1f; // change text every 0.1 seconds

    void Update()
    {
        timeAccumulator += Time.unscaledDeltaTime;

        if (timeAccumulator >= TARGET_FRAME_WINDOW)
        {
            fpsText.text = "FPS: " + agent.measuredFPS.ToString() + "\n" +
                           "Quality: " + agent.quality.ToString() + "\n" +
                           "Actions: " + agent.actionCount.ToString();

            timeAccumulator = 0f;
        }
    }
}
