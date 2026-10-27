// Mesh intersections for independent control-unit calibration. Scene changes
// are transient; this editor job never saves assets or changes the orbit.
using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering.Universal;

public static class LightingSeedGeometry
{
    [Serializable] public class Pixel { public float u, v; }
    [Serializable] public class Pixels { public Pixel[] points; }
    [Serializable] public class Hit
    {
        public float u, v;
        public Vector3 worldM;
        public string mesh;
    }
    [Serializable] public class Geometry
    {
        public int schemaVersion = 1, width = 1024, height = 1024;
        public string coordinates = "right_handed_y_up_opencv_c2w";
        public string method = "Exact static MeshFilter triangle intersections; original approximate colliders disabled";
        public string limitations = "Opaque mesh geometry; alpha textures and skinned meshes are not ray-traced; RANSAC must reject mismatches";
        public string calibrationSha256;
        public int meshCount;
        public Hit[] points;
    }
    static string Arg(string key)
    {
        var args=Environment.GetCommandLineArgs(); int i=Array.IndexOf(args,key);
        return i>=0 && i+1<args.Length ? args[i+1] : null;
    }
    public static void Run()
    {
        try { Capture(); EditorApplication.Exit(0); }
        catch(Exception e) { Debug.LogException(e); EditorApplication.Exit(1); }
    }
    static void Capture()
    {
        string output=Arg("-geometryOutput") ?? throw new Exception("Missing output");
        if(Directory.Exists(output)) throw new Exception("Refusing overwrite");
        byte[] calibrationBytes=File.ReadAllBytes(Arg("-geometryCalibration"));
        var c=JsonUtility.FromJson<LightingOrbitCapture.Calibration>(System.Text.Encoding.UTF8.GetString(calibrationBytes));
        var pixels=JsonUtility.FromJson<Pixels>(File.ReadAllText(Arg("-geometryPixels")));
        EditorSceneManager.OpenScene(c.scene,OpenSceneMode.Single);
        foreach(var b in UnityEngine.Object.FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None))
            if(!(b is UniversalAdditionalCameraData) && !(b is UniversalAdditionalLightData)) b.enabled=false;
        foreach(var a in UnityEngine.Object.FindObjectsByType<Animator>(FindObjectsSortMode.None)) a.enabled=false;
        foreach(var p in UnityEngine.Object.FindObjectsByType<ParticleSystem>(FindObjectsSortMode.None)) p.Stop(true,ParticleSystemStopBehavior.StopEmittingAndClear);
        foreach(var b in UnityEngine.Object.FindObjectsByType<Rigidbody>(FindObjectsSortMode.None)) b.isKinematic=true;
        var lights=UnityEngine.Object.FindObjectsByType<Light>(FindObjectsSortMode.None);
        var sun=lights.Single(l=>l.type==LightType.Directional);
        foreach(var l in lights) l.enabled=l==sun;
        RenderSettings.sun=sun;
        var camera=Camera.main ?? throw new Exception("Missing MainCamera");
        camera.enabled=false; camera.aspect=1; camera.fieldOfView=c.verticalFovDeg;
        camera.transform.position=c.originUnity+new Vector3(0,c.eyeHeightM,-c.radiusM);
        camera.transform.LookAt(c.targetUnity,Vector3.up);
        var urp=camera.GetUniversalAdditionalCameraData(); urp.renderPostProcessing=false; urp.cameraStack.Clear();
        Directory.CreateDirectory(output);
        var rt=new RenderTexture(c.width,c.height,24,RenderTextureFormat.ARGB32);
        rt.Create(); camera.targetTexture=rt; camera.Render(); RenderTexture.active=rt;
        var texture=new Texture2D(c.width,c.height,TextureFormat.RGB24,false);
        texture.ReadPixels(new Rect(0,0,c.width,c.height),0,0); texture.Apply(); RenderTexture.active=null;
        File.WriteAllBytes(Path.Combine(output,"ref.png"),texture.EncodeToPNG());
        foreach(var old in UnityEngine.Object.FindObjectsByType<Collider>(FindObjectsSortMode.None)) old.enabled=false;
        int meshes=0;
        foreach(var f in UnityEngine.Object.FindObjectsByType<MeshFilter>(FindObjectsSortMode.None))
        {
            var r=f.GetComponent<Renderer>();
            if(f.sharedMesh==null || r==null || !r.enabled || !r.gameObject.activeInHierarchy) continue;
            var collider=f.gameObject.AddComponent<MeshCollider>(); collider.sharedMesh=f.sharedMesh;
            collider.convex=false; meshes++;
        }
        Physics.SyncTransforms();
        float focal=c.height/(2*Mathf.Tan(c.verticalFovDeg*Mathf.Deg2Rad/2));
        var hits=new System.Collections.Generic.List<Hit>();
        foreach(var p in pixels.points)
        {
            float x=(p.u-c.width/2f)/focal,y=(p.v-c.height/2f)/focal;
            Vector3 direction=(camera.transform.right*x-camera.transform.up*y+camera.transform.forward).normalized;
            if(Physics.Raycast(new Ray(camera.transform.position,direction),out RaycastHit h,100,-1,QueryTriggerInteraction.Ignore))
            {
                Vector3 delta=h.point-c.originUnity;
                hits.Add(new Hit {u=p.u,v=p.v,worldM=new Vector3(delta.x,delta.y,-delta.z),mesh=h.collider.name});
            }
        }
        var result=new Geometry {meshCount=meshes,points=hits.ToArray(),
            calibrationSha256=BitConverter.ToString(SHA256.Create().ComputeHash(calibrationBytes)).Replace("-","").ToLowerInvariant()};
        File.WriteAllText(Path.Combine(output,"seed_geometry.json"),JsonUtility.ToJson(result,true));
        camera.targetTexture=null; rt.Release();
        UnityEngine.Object.DestroyImmediate(rt); UnityEngine.Object.DestroyImmediate(texture);
        Debug.Log($"[LightingSeedGeometry] {hits.Count} points from {meshes} meshes");
    }
}
