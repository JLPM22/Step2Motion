using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ResetMaterialsOnDestroy : MonoBehaviour
{
    private void OnDestroy()
    {
        Renderer[] renderers = GetComponentsInChildren<Renderer>();
        foreach (Renderer renderer in renderers)
        {
            if (renderer.name != "Base")
            {
                renderer.sharedMaterial.color = new Color(0.8f, 0.8f, 0.8f);
            }
        }
    }
}
