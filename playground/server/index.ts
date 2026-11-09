import { createPlaygroundServer } from './app.js';
import { options } from './config.js';

const settings = options();
const app = createPlaygroundServer(settings);
app.server.listen(settings.port, settings.host, () => console.log(`Playground: http://${settings.host}:${settings.port}`));
app.server.on('error', error => { console.error(error.message); process.exitCode = 1; });
for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, () => { void app.close().then(() => process.exit(0)); });
