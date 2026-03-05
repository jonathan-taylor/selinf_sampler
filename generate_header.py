import os

# downloaded: https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201

def generate_sobol_header(input_file, output_file, max_dims=21201):
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        return

    with open(input_file, 'r') as f:
        lines = f.readlines()
        # Joe-Kuo file usually has a header row like "d s a m_i"
        if lines[0].strip().startswith('d'):
            lines = lines[1:]

    with open(output_file, 'w') as f:
        f.write("#ifndef SOBOL_DATA_HPP\n#define SOBOL_DATA_HPP\n\n")
        f.write("#include <cstdint>\n\n")
        f.write("struct SobolParams {\n")
        f.write("    uint32_t d, s, a;\n")
        f.write("    uint32_t m[18]; // Max degree in Joe-Kuo is 18\n")
        f.write("};\n\n")
        f.write(f"static const SobolParams JOE_KUO_PARAMS[{min(len(lines), max_dims)}] = {{\n")

        for i, line in enumerate(lines[:max_dims]):
            parts = list(map(int, line.split()))
            d, s, a = parts[0], parts[1], parts[2]
            m_values = parts[3:]
            
            # Pad m_values with 0s to reach exactly size 18 for the struct
            m_padded = m_values + [0] * (18 - len(m_values))
            m_list_str = ", ".join(map(str, m_padded))
            
            f.write(f"    {{ {d}, {s}, {a}, {{{m_list_str}}} }},\n")

        f.write("};\n\n#endif // SOBOL_DATA_HPP\n")

if __name__ == "__main__":
    # Ensure the file name matches exactly what you downloaded
    generate_sobol_header('new-joe-kuo-6.21201', 'sobol_data.hpp')
    print("Successfully generated sobol_data.hpp")
