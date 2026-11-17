namespace Lpc.Cli;

/// <summary>Minimal "--name value" / "--flag" parser.</summary>
public sealed class Args
{
    readonly Dictionary<string, string?> options = new(StringComparer.OrdinalIgnoreCase);
    public List<string> Positional { get; } = [];

    public Args(IEnumerable<string> args)
    {
        var list = args.ToList();
        for (int i = 0; i < list.Count; i++)
        {
            var a = list[i];
            if (!a.StartsWith("--")) { Positional.Add(a); continue; }
            var name = a[2..];
            string? value = null;
            int eq = name.IndexOf('=');
            if (eq >= 0) { value = name[(eq + 1)..]; name = name[..eq]; }
            else if (i + 1 < list.Count && !list[i + 1].StartsWith("--")) value = list[++i];
            options[name] = value;
        }
    }

    public bool Has(string name) => options.ContainsKey(name);
    public string Get(string name, string fallback) => options.TryGetValue(name, out var v) && v != null ? v : fallback;
    public string? Get(string name) => options.TryGetValue(name, out var v) ? v : null;
    public int Int(string name, int fallback) => int.TryParse(Get(name), out var v) ? v : fallback;
    public double Double(string name, double fallback) =>
        double.TryParse(Get(name), System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : fallback;
}
