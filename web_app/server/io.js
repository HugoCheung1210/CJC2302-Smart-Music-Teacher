const socketIo = require('socket.io');
let io = null;

function initialize(server) {
  io = socketIo(server, {
    cors: {
      origin: '*',
    }
  });
  return io;
}

module.exports = {
  initialize,
  getIo: () => {
    if (!io) {
      throw new Error("Socket.io not initialized!");
    }
    return io;
  }
};