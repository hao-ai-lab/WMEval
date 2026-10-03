// Patch a copied Mono Unity player so GardenGTCapture starts at an explicit pose.
// This is a temporary, auditable bridge when the Unity Editor license is absent;
// the authoritative GardenGTCapture.cs source supports the equivalent CLI args.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Mono.Cecil;
using Mono.Cecil.Cil;

public static class PatchGardenCapturePose
{
    public static int Main(string[] args)
    {
        if (args.Length != 6)
        {
            Console.Error.WriteLine("usage: patcher INPUT.dll OUTPUT.dll X Y Z YAW_DEG");
            return 2;
        }

        float x = Parse(args[2]);
        float y = Parse(args[3]);
        float z = Parse(args[4]);
        float yaw = Parse(args[5]);

        using (var assembly = AssemblyDefinition.ReadAssembly(args[0]))
        {
            ModuleDefinition module = assembly.MainModule;
            TypeDefinition capture = module.Types.Single(t => t.Name == "GardenGTCapture");
            MethodDefinition start = capture.Methods.Single(m => m.Name == "Start" && !m.HasParameters);
            FieldDefinition startPos = capture.Fields.Single(f => f.Name == "_startPos");
            FieldDefinition startRot = capture.Fields.Single(f => f.Name == "_startRot");
            MethodReference vectorCtor = FindMethodReference(
                module, "UnityEngine.Vector3", ".ctor", 3);
            MethodReference quaternionEuler = FindMethodReference(
                module, "UnityEngine.Quaternion", "Euler", 3);

            ILProcessor il = start.Body.GetILProcessor();
            Instruction target = start.Body.Instructions.Last(i => i.OpCode == OpCodes.Ret);
            foreach (Instruction instruction in new[]
            {
                il.Create(OpCodes.Ldarg_0),
                il.Create(OpCodes.Ldc_R4, x),
                il.Create(OpCodes.Ldc_R4, y),
                il.Create(OpCodes.Ldc_R4, z),
                il.Create(OpCodes.Newobj, vectorCtor),
                il.Create(OpCodes.Stfld, startPos),
                il.Create(OpCodes.Ldarg_0),
                il.Create(OpCodes.Ldc_R4, 0f),
                il.Create(OpCodes.Ldc_R4, yaw),
                il.Create(OpCodes.Ldc_R4, 0f),
                il.Create(OpCodes.Call, quaternionEuler),
                il.Create(OpCodes.Stfld, startRot),
            })
            {
                il.InsertBefore(target, instruction);
            }

            assembly.Write(args[1]);
        }

        Console.WriteLine(
            string.Format(
                CultureInfo.InvariantCulture,
                "patched GardenGTCapture pose to ({0}, {1}, {2}), yaw {3}",
                x, y, z, yaw));
        return 0;
    }

    private static float Parse(string raw)
    {
        return float.Parse(raw, NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    private static IEnumerable<TypeDefinition> AllTypes(ModuleDefinition module)
    {
        var pending = new Stack<TypeDefinition>(module.Types);
        while (pending.Count > 0)
        {
            TypeDefinition current = pending.Pop();
            yield return current;
            foreach (TypeDefinition nested in current.NestedTypes) pending.Push(nested);
        }
    }

    private static MethodReference FindMethodReference(
        ModuleDefinition module, string declaringType, string methodName, int parameterCount)
    {
        foreach (TypeDefinition type in AllTypes(module))
        {
            foreach (MethodDefinition method in type.Methods.Where(m => m.HasBody))
            {
                foreach (Instruction instruction in method.Body.Instructions)
                {
                    var candidate = instruction.Operand as MethodReference;
                    if (candidate != null && candidate.DeclaringType.FullName == declaringType &&
                        candidate.Name == methodName && candidate.Parameters.Count == parameterCount)
                    {
                        return candidate;
                    }
                }
            }
        }
        throw new InvalidOperationException(
            "could not locate method reference " + declaringType + "::" + methodName);
    }
}
