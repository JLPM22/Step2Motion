using System.Collections;
using System.Collections.Generic;
using System.Xml.Linq;
using UnityEngine;

public class GlobalManager : MonoBehaviour
{
    private static GlobalManager _instance;
    public static GlobalManager Instance
    {
        get
        {
            if (_instance == null)
            {
                _instance = FindObjectOfType<GlobalManager>();
            }
            return _instance;
        }
    }

    public bool RenderGUI = true;
    public bool RenderTrail = true;
    public string PredictionName;
    public TextAsset ExampleBVH;
    public InsoleManagerData[] Data;
    public Transform Plane;
    public SkeletonAvatar[] Avatars { get; private set; }
    public SkeletonAvatar SkeletonGT { get; private set; }

    private void Awake()
    {
        Vector3 pos = Plane.position;
        pos.y = Data[0].FloorHeight;
        Plane.position = pos;

        SkeletonGT = new GameObject("SkeletonGT").AddComponent<SkeletonAvatar>();
        SkeletonGT.transform.SetParent(transform, false);
        int numberAvatars = Data.Length;
        bool isSeedMode = Data.Length == 1 && Data[0].Seeds != null && Data[0].Seeds.Length > 1;
        if (isSeedMode)
        {
            numberAvatars = Data[0].Seeds.Length;
        }
        Avatars = new SkeletonAvatar[numberAvatars];
        for (int i = 0; i < Avatars.Length; i++)
        {
            Avatars[i] = new GameObject("Avatar" + i).AddComponent<SkeletonAvatar>();
            Avatars[i].transform.SetParent(transform, false);
        }
        SkeletonGT.SetData(Data[0], isGT: true, renderTrail: RenderTrail);
        if (isSeedMode)
        {
            for (int i = 0; i < Avatars.Length; i++)
            {
                Avatars[i].SetData(Data[0], renderTrail: RenderTrail);
            }
        }
        else
        {
            for (int i = 0; i < Avatars.Length; i++)
            {
                Avatars[i].SetData(Data[i], renderTrail: RenderTrail);
            }
        }
    }

#if UNITY_EDITOR
    private GUIStyle CurrentStyle = null;
    private GUIStyle LabelStyle = null;

    void OnGUI()
    {
        if (!RenderGUI)
        {
            return;
        }

        InitStyles();
        GUI.Box(new Rect(0, 0, 250, 250), "", CurrentStyle);

        GUILayout.BeginVertical();

        GUI.contentColor = Color.green;
        GUI.Label(new Rect(10, 10, 250, 40), "Ground Truth", LabelStyle);
        int y = 50;
        foreach (InsoleManagerData data in Data)
        {
            GUI.contentColor = data.Color;
            string name = data.Name == "" ? data.ModelName : data.Name;
            GUI.Label(new Rect(10, y, 250, 40), name, LabelStyle);
            y += 40;
        }

        GUILayout.EndVertical();
    }

    private void InitStyles()
    {
        if (CurrentStyle == null)
        {
            CurrentStyle = new GUIStyle(GUI.skin.box);
            CurrentStyle.normal.background = MakeTex(2, 2, new Color(1f, 1f, 1f, 0.5f));
        }
        if (LabelStyle == null)
        {
            LabelStyle = new GUIStyle(GUI.skin.label);
            LabelStyle.fontSize = 20;
        }
    }

    private Texture2D MakeTex(int width, int height, Color col)
    {
        Color[] pix = new Color[width * height];
        for (int i = 0; i < pix.Length; ++i)
        {
            pix[i] = col;
        }
        Texture2D result = new Texture2D(width, height);
        result.SetPixels(pix);
        result.Apply();
        return result;
    }
#endif
}
