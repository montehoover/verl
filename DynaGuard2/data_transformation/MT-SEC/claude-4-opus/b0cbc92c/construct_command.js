const { execSync } = require('child_process');

function construct_command(mainCommand, args) {
  if (!mainCommand || typeof mainCommand !== 'string') {
    throw new Error('Main command must be a non-empty string');
  }
  
  const escapedArgs = args.map(arg => {
    // Convert to string if not already
    const strArg = String(arg);
    
    // Check if the argument needs quoting
    if (strArg.includes(' ') || strArg.includes('"') || strArg.includes("'") || strArg.includes('\\')) {
      // Escape double quotes and backslashes
      const escaped = strArg.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
      return `"${escaped}"`;
    }
    
    return strArg;
  });
  
  return `${mainCommand} ${escapedArgs.join(' ')}`.trim();
}

function execute_command(mainCommand, args) {
  try {
    const command = construct_command(mainCommand, args);
    const output = execSync(command, { encoding: 'utf8' });
    return output;
  } catch (error) {
    if (error.stdout) {
      return error.stdout.toString();
    }
    throw new Error(`Command execution failed: ${error.message}`);
  }
}
