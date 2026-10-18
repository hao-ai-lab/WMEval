// Copy into Assets/Editor in an isolated SunRisesAndSets project. Never save
// the scene: all overrides exist only in this editor process.
using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering.Universal;

public static class LightingOrbitCapture
{
    [Serializable] public class Calibration
    {
        public string scene, unityVersion, tableSelection, targetSelection, floorSelection;
        public Vector3 tableBoundsCenter, tableBoundsSize, originUnity, targetUnity;
        public float radiusM = 2, eyeHeightM = 1.6f, durationSec = 10, verticalFovDeg;
        public int width = 1024, height = 1024, fps = 12;
        public string localToUnity = "diag(1,1,-1) plus originUnity; local RH +Y up; initial local +Z";
        public LightRecord[] lights;
        public string[] disabledBehaviours;
    }
    [Serializable] public class LightRecord
    {
        public string name, type;
        public Vector3 position, euler, forward;
        public Color color;
        public float intensity, colorTemperature;
        public bool enabled, useColorTemperature;
    }
    [Serializable] public class Frame
    {
        public int frameIndex;
        public double canonicalTimeSec;
        public Vector3 positionUnity, rightUnity, upUnity, forwardUnity;
        public string staticStateSha256;
        public LightRecord sun;
    }
    [Serializable] public class Dependency { public string path, sha256; }
    [Serializable] public class Dependencies { public Dependency[] files; }
    static string Hash(byte[] bytes) => BitConverter.ToString(SHA256.Create().ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
    static LightRecord LightState(Light l) => new LightRecord {
        name=l.name, type=l.type.ToString(), position=l.transform.position,
        euler=l.transform.eulerAngles, forward=l.transform.forward, color=l.color,
        intensity=l.intensity, colorTemperature=l.colorTemperature,
        useColorTemperature=l.useColorTemperature, enabled=l.enabled
    };
    static string StaticState()
    {
        var rows = UnityEngine.Object.FindObjectsByType<Renderer>(FindObjectsSortMode.None)
            .Select(r => r.name+"|"+r.transform.localToWorldMatrix.ToString("R")+"|"+r.enabled)
            .OrderBy(s => s);
        return Hash(Encoding.UTF8.GetBytes(string.Join("\n", rows)));
    }
    static string Arg(string key)
    {
        var args = Environment.GetCommandLineArgs();
        int i = Array.IndexOf(args, key);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }
    public static void Run()
    {
        try { Capture(); EditorApplication.Exit(0); }
        catch (Exception e) { Debug.LogException(e); EditorApplication.Exit(1); }
    }
    static void Capture()
    {
        string output = Arg("-orbitOutput") ?? throw new Exception("Missing -orbitOutput");
        if (Directory.Exists(output)) throw new Exception("Refusing to overwrite " + output);
        string scene = "Assets/Scenes/SampleScene.unity";
        EditorSceneManager.OpenScene(scene, OpenSceneMode.Single);
        var table = GameObject.Find("Table23") ?? throw new Exception("Expected Table23");
        var floor = GameObject.Find("Plane") ?? throw new Exception("Expected Plane");
        var bounds = table.GetComponentsInChildren<Renderer>().First().bounds;
        foreach (var r in table.GetComponentsInChildren<Renderer>()) bounds.Encapsulate(r.bounds);
        // Mesh bounds, not an approximate physics collider, define the top.
        float floorY = floor.transform.position.y;
        Vector3 origin = new Vector3(bounds.center.x, floorY, bounds.center.z);
        Vector3 target = new Vector3(bounds.center.x, bounds.max.y, bounds.center.z);
        var camera = Camera.main ?? throw new Exception("Expected MainCamera");
        var scripts = UnityEngine.Object.FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None);
        var disabled = scripts.Where(b => b != null && b.enabled && !(b is UniversalAdditionalCameraData)
            && !(b is UniversalAdditionalLightData)).ToArray();
        foreach (var b in disabled) b.enabled = false;
        foreach (var a in UnityEngine.Object.FindObjectsByType<Animator>(FindObjectsSortMode.None)) a.enabled = false;
        foreach (var p in UnityEngine.Object.FindObjectsByType<ParticleSystem>(FindObjectsSortMode.None)) p.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
        foreach (var b in UnityEngine.Object.FindObjectsByType<Rigidbody>(FindObjectsSortMode.None)) b.isKinematic = true;
        var lights = UnityEngine.Object.FindObjectsByType<Light>(FindObjectsSortMode.None);
        var sun = lights.Single(l => l.type == LightType.Directional);
        foreach (var l in lights) l.enabled = l == sun;
        RenderSettings.sun = sun;
        camera.enabled = false;
        camera.aspect = 1;
        var urp = camera.GetUniversalAdditionalCameraData();
        urp.renderPostProcessing = false;
        urp.cameraStack.Clear();
        var calibration = new Calibration {
            scene = scene, unityVersion = Application.unityVersion,
            tableSelection = "GameObject exact name Table23; aggregate Renderer.bounds",
            targetSelection = "table mesh bounds centre x/z and max.y",
            floorSelection = "GameObject Plane transform.position.y",
            tableBoundsCenter = bounds.center, tableBoundsSize = bounds.size,
            originUnity = origin, targetUnity = target, verticalFovDeg = camera.fieldOfView,
            disabledBehaviours = disabled.Select(b => b.GetType().FullName).Distinct().OrderBy(s => s).ToArray(),
            lights = lights.OrderBy(l => l.name).Select(LightState).ToArray()
        };
        Directory.CreateDirectory(output);
        Directory.CreateDirectory(Path.Combine(output, "frames"));
        File.WriteAllText(Path.Combine(output,"calibration.json"), JsonUtility.ToJson(calibration, true));
        var dependencies = AssetDatabase.GetDependencies(scene, true).Where(File.Exists).OrderBy(s => s)
            .Select(p => new Dependency {path=p, sha256=Hash(File.ReadAllBytes(p))}).ToArray();
        File.WriteAllText(Path.Combine(output,"scene_dependencies.json"), JsonUtility.ToJson(new Dependencies {files=dependencies}, true));
        var rt = new RenderTexture(calibration.width, calibration.height, 24, RenderTextureFormat.ARGB32);
        rt.Create(); camera.targetTexture = rt;
        var texture = new Texture2D(calibration.width, calibration.height, TextureFormat.RGB24, false);
        using (var frames = new StreamWriter(Path.Combine(output,"frames.jsonl")))
        {
            for (int i=0; i<=calibration.durationSec*calibration.fps; i++)
            {
                double t = (double)i/calibration.fps;
                double angle = 2*Math.PI*t/calibration.durationSec;
                camera.transform.position = origin + new Vector3(
                    calibration.radiusM*(float)Math.Sin(angle), calibration.eyeHeightM,
                    -calibration.radiusM*(float)Math.Cos(angle));
                camera.transform.LookAt(target, Vector3.up);
                camera.Render(); RenderTexture.active=rt;
                texture.ReadPixels(new Rect(0,0,rt.width,rt.height),0,0); texture.Apply();
                RenderTexture.active=null;
                var png=texture.EncodeToPNG();
                File.WriteAllBytes(Path.Combine(output,"frames",$"frame_{i:D6}.png"),png);
                if (i==0) File.WriteAllBytes(Path.Combine(output,"ref.png"),png);
                frames.WriteLine(JsonUtility.ToJson(new Frame {
                    frameIndex=i, canonicalTimeSec=t, positionUnity=camera.transform.position,
                    rightUnity=camera.transform.right, upUnity=camera.transform.up,
                    forwardUnity=camera.transform.forward, sun=LightState(sun), staticStateSha256=StaticState()
                }));
            }
        }
        camera.targetTexture=null; rt.Release();
        UnityEngine.Object.DestroyImmediate(rt); UnityEngine.Object.DestroyImmediate(texture);
        Debug.Log("[LightingOrbitCapture] Complete: " + output);
    }
}
